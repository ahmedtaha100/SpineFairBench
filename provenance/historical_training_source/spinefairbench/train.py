from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from spinefairbench.config.schemas import load_config
from spinefairbench.data.dataloader import create_dataloader
from spinefairbench.models.cycle_diffusion import CycleDiffusionModel
from spinefairbench.utils.device import get_device
from spinefairbench.utils.logging import setup_logging
from spinefairbench.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def _create_run_dir(base: Path, name: str = "train") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _init_wandb(config: dict, run_dir: Path) -> bool:
    try:
        import wandb

        wandb.init(project="spinefairbench", config=config, dir=str(run_dir))
        return True
    except Exception as e:
        logger.warning("W&B init failed: %s. Continuing without tracking.", e)
        return False


def _save_sample(decoded: torch.Tensor, path: Path) -> None:
    img = decoded[0].detach().cpu().clamp(-1, 1).float().numpy()
    if img.ndim == 3:
        img = img[0]
    img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _amp_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _log_wandb(metrics: dict, step: int) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def run_training(args: argparse.Namespace) -> None:
    setup_logging()

    config_path = Path(args.config)
    config = load_config(config_path)
    tc = config.training
    dc = config.data

    seed_everything(tc.seed)
    device = get_device()

    data_root = Path(dc.root_dir)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_root}. "
            "Download the VinDr-SpineXR dataset before training."
        )

    run_dir = _create_run_dir(Path("outputs"), "train")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run directory: %s", run_dir)
    _init_wandb(config.model_dump(), run_dir)

    model = CycleDiffusionModel(
        base_model_name=tc.base_model,
        lora_rank=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        lambda_adversarial=tc.lambda_adversarial,
        lambda_kl=tc.lambda_kl,
        lambda_cycle=0.0,
    )

    model.load_pretrained(device)
    model.to(device)

    gen_params = (
        list(model.demographic_encoder.parameters())
        + [p for p in model._unet.parameters() if p.requires_grad]
    )
    disc_params = list(model.discriminator.parameters())

    optimizer_g = AdamW(gen_params, lr=tc.learning_rate)
    optimizer_d = AdamW(disc_params, lr=tc.learning_rate)
    scheduler_g = CosineAnnealingWarmRestarts(optimizer_g, T_0=tc.warmup_steps, T_mult=2)

    use_amp = tc.mixed_precision != "no"
    amp_dtype = _amp_dtype(tc.mixed_precision)
    device_type = device.type
    scaler = torch.amp.GradScaler(enabled=use_amp and tc.mixed_precision == "fp16")

    train_loader = create_dataloader(
        data_root=dc.root_dir,
        annotations_csv=dc.annotations_csv,
        split="train",
        batch_size=tc.batch_size,
        num_workers=dc.num_workers,
        image_size=dc.image_size,
        normalize_range="-1_1",
        train_ratio=dc.train_split,
        val_ratio=dc.val_split,
        test_ratio=dc.test_split,
        seed=tc.seed,
        cache_dir=dc.cache_dir,
        source_filter=dc.source_filter,
    )
    val_loader = create_dataloader(
        data_root=dc.root_dir,
        annotations_csv=dc.annotations_csv,
        split="val",
        batch_size=tc.batch_size,
        num_workers=dc.num_workers,
        image_size=dc.image_size,
        normalize_range="-1_1",
        train_ratio=dc.train_split,
        val_ratio=dc.val_split,
        test_ratio=dc.test_split,
        seed=tc.seed,
        cache_dir=dc.cache_dir,
        source_filter=dc.source_filter,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = model.load_checkpoint(resume_path, device)
        if "optimizer_g" in checkpoint:
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if "optimizer_d" in checkpoint:
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        logger.info("Resumed from epoch %d, step %d", start_epoch, global_step)

    logger.info("=== Stage 1: LDM Pre-training ===")
    model.set_stage(1)

    for epoch in range(start_epoch, tc.num_epochs_stage1):
        model.train()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}/{tc.num_epochs_stage1}")
        for batch in pbar:
            image = batch["image"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)

            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                step_output = model.forward_step(image, age, sex)
                disc_losses = model.compute_discriminator_loss(step_output, image)

            optimizer_d.zero_grad()
            scaler.scale(disc_losses["disc_total"]).backward()
            scaler.step(optimizer_d)
            scaler.update()

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                step_output = model.forward_step(image, age, sex)
                gen_losses = model.compute_generator_loss(step_output, image)

            loss_g = gen_losses["total"] / tc.gradient_accumulation_steps
            scaler.scale(loss_g).backward()

            if (global_step + 1) % tc.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(gen_params, tc.max_grad_norm)
                scaler.step(optimizer_g)
                scaler.update()
                scheduler_g.step()
                optimizer_g.zero_grad()

            epoch_loss_g += gen_losses["total"].item()
            epoch_loss_d += disc_losses["disc_total"].item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({
                "g_loss": f"{gen_losses['total'].item():.4f}",
                "d_loss": f"{disc_losses['disc_total'].item():.4f}",
            })

            _log_wandb({
                "train/loss_g_total": gen_losses["total"].item(),
                "train/loss_l1": gen_losses["l1"].item(),
                "train/loss_d_total": disc_losses["disc_total"].item(),
                "train/lr": optimizer_g.param_groups[0]["lr"],
                "train/epoch": epoch,
            }, step=global_step)

            if global_step % tc.sample_every == 0:
                with torch.no_grad():
                    sample_decoded = model.decode_latent(step_output["latent"])
                _save_sample(sample_decoded, sample_dir / f"sample_step_{global_step}.png")

            if global_step % tc.checkpoint_every == 0:
                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                model.save_checkpoint(
                    ckpt_path,
                    optimizer_g=optimizer_g.state_dict(),
                    optimizer_d=optimizer_d.state_dict(),
                    epoch=epoch,
                    global_step=global_step,
                )

        if global_step % tc.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(gen_params, tc.max_grad_norm)
            scaler.step(optimizer_g)
            scaler.update()
            scheduler_g.step()
            optimizer_g.zero_grad()

        avg_g = epoch_loss_g / max(num_batches, 1)
        avg_d = epoch_loss_d / max(num_batches, 1)
        logger.info(
            "Epoch %d/%d complete. G loss: %.4f, D loss: %.4f",
            epoch + 1, tc.num_epochs_stage1, avg_g, avg_d,
        )

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_image = val_batch["image"].to(device)
                val_age = val_batch["age"].to(device)
                val_sex = val_batch["sex"].to(device)
                if val_image.shape[1] == 1:
                    val_image = val_image.repeat(1, 3, 1, 1)
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    val_output = model.forward_step(val_image, val_age, val_sex)
                    val_gen_losses = model.compute_generator_loss(val_output, val_image)
                val_loss_sum += val_gen_losses["total"].item()
                val_batches += 1
        avg_val = val_loss_sum / max(val_batches, 1)
        logger.info("Epoch %d/%d val loss: %.4f", epoch + 1, tc.num_epochs_stage1, avg_val)
        _log_wandb({"val/loss_g_total": avg_val, "val/epoch": epoch}, step=global_step)

        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        model.save_checkpoint(
            ckpt_path,
            optimizer_g=optimizer_g.state_dict(),
            optimizer_d=optimizer_d.state_dict(),
            epoch=epoch + 1,
            global_step=global_step,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "best.pt"
            model.save_checkpoint(
                best_path,
                optimizer_g=optimizer_g.state_dict(),
                optimizer_d=optimizer_d.state_dict(),
                epoch=epoch + 1,
                global_step=global_step,
            )
            logger.info("New best val loss: %.4f (saved to %s)", best_val_loss, best_path)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement for %d epoch(s) (best: %.4f, current: %.4f)",
                epochs_without_improvement, best_val_loss, avg_val,
            )

        if (
            tc.early_stopping
            and epoch + 1 >= tc.early_stopping_min_epochs
            and epochs_without_improvement >= tc.early_stopping_patience
        ):
            logger.info(
                "Early stopping triggered at epoch %d (patience %d, min epochs %d)",
                epoch + 1, tc.early_stopping_patience, tc.early_stopping_min_epochs,
            )
            break

    logger.info("=== Stage 2: Cycle Consistency Fine-tuning ===")
    model.set_stage(2, lambda_cycle=tc.lambda_cycle)
    model.train()
    optimizer_g.zero_grad()

    best_val_loss_s2 = float("inf")
    iters_without_improvement_s2 = 0
    s2_iteration = 0

    for _iteration in tqdm(range(tc.num_iterations_stage2), desc="Stage 2"):
        for batch in train_loader:
            image = batch["image"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)

            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)

            flipped_sex = 1 - sex

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                cycle_output = model.cycle_forward(
                    image,
                    source_age=age,
                    source_sex=sex,
                    target_age=age,
                    target_sex=flipped_sex,
                )
                disc_losses = model.compute_discriminator_loss(cycle_output, image)

            optimizer_d.zero_grad()
            scaler.scale(disc_losses["disc_total"]).backward()
            scaler.step(optimizer_d)
            scaler.update()

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                cycle_output = model.cycle_forward(
                    image,
                    source_age=age,
                    source_sex=sex,
                    target_age=age,
                    target_sex=flipped_sex,
                )
                gen_losses = model.compute_generator_loss(
                    cycle_output,
                    image,
                    original_latent=cycle_output["latent_a"],
                    reconstructed_latent=cycle_output["latent_a_prime"],
                )

            scaler.scale(gen_losses["total"]).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(gen_params, tc.max_grad_norm)
            scaler.step(optimizer_g)
            scaler.update()
            optimizer_g.zero_grad()
            global_step += 1
            s2_iteration += 1

            _log_wandb({
                "train_s2/loss_g_total": gen_losses["total"].item(),
                "train_s2/loss_cycle": gen_losses.get("cycle", torch.tensor(0.0)).item(),
                "train_s2/loss_d_total": disc_losses["disc_total"].item(),
            }, step=global_step)

            break

        if s2_iteration % 50 == 0 and s2_iteration > 0:
            model.eval()
            val_loss_s2 = 0.0
            val_batches_s2 = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    vi = val_batch["image"].to(device)
                    va = val_batch["age"].to(device)
                    vs = val_batch["sex"].to(device)
                    if vi.shape[1] == 1:
                        vi = vi.repeat(1, 3, 1, 1)
                    fs = 1 - vs
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        co = model.cycle_forward(vi, source_age=va, source_sex=vs, target_age=va, target_sex=fs)
                        vl = model.compute_generator_loss(co, vi, original_latent=co["latent_a"], reconstructed_latent=co["latent_a_prime"])
                    val_loss_s2 += vl["total"].item()
                    val_batches_s2 += 1
                    if val_batches_s2 >= 20:
                        break
            avg_val_s2 = val_loss_s2 / max(val_batches_s2, 1)
            logger.info("Stage 2 iter %d val loss: %.4f", s2_iteration, avg_val_s2)
            model.train()

            if avg_val_s2 < best_val_loss_s2:
                best_val_loss_s2 = avg_val_s2
                iters_without_improvement_s2 = 0
                model.save_checkpoint(
                    checkpoint_dir / "best_s2.pt",
                    optimizer_g=optimizer_g.state_dict(),
                    optimizer_d=optimizer_d.state_dict(),
                    epoch=tc.num_epochs_stage1,
                    global_step=global_step,
                )
            else:
                iters_without_improvement_s2 += 50

            if (
                tc.early_stopping
                and s2_iteration >= tc.early_stopping_min_iterations_stage2
                and iters_without_improvement_s2 >= tc.early_stopping_patience_stage2
            ):
                logger.info("Stage 2 early stopping at iteration %d", s2_iteration)
                break

    final_path = checkpoint_dir / "final.pt"
    model.save_checkpoint(
        final_path,
        optimizer_g=optimizer_g.state_dict(),
        optimizer_d=optimizer_d.state_dict(),
        epoch=tc.num_epochs_stage1,
        global_step=global_step,
    )

    latest_path = Path("checkpoints/latest.pt")
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    best_s2 = checkpoint_dir / "best_s2.pt"
    best_s1 = checkpoint_dir / "best.pt"
    if best_s2.exists():
        import shutil
        shutil.copy2(best_s2, latest_path)
        logger.info("Using best stage 2 checkpoint as latest: %s", latest_path)
    elif best_s1.exists():
        import shutil
        shutil.copy2(best_s1, latest_path)
        logger.info("Using best stage 1 checkpoint as latest: %s", latest_path)
    else:
        model.save_checkpoint(latest_path)

    logger.info("Training complete. Final checkpoint: %s", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleDiffusion model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    run_training(parser.parse_args())
