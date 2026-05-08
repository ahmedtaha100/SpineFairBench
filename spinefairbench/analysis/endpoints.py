from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional analysis dependency
    np = None

try:
    from scipy import stats
except ModuleNotFoundError:  # pragma: no cover - optional analysis dependency
    stats = None


@dataclass
class PairMetric:
    source_id: str
    pair_id: str
    model: str
    tier: str
    recommendation_changed: float
    diagnostic_consistency: float
    severity_disparity: float
    confidence_disparity: float


# Structured analysis helper scale for parsed severity fields only. This is not
# the free-text severity-language helper in spinefairbench.metrics.
STRUCTURED_SEVERITY_MAP_0_TO_3 = {
    "normal": 0.0,
    "mild": 1.0,
    "moderate": 2.0,
    "severe": 3.0,
}


def _require_numeric_stack() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if stats is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "spinefairbench.analysis.endpoints requires optional analysis "
            f"dependencies not installed on the reviewer path: {', '.join(missing)}"
        )


def _row_model(row: dict[str, Any]) -> str:
    return str(row.get("requested_model") or row.get("model") or "")


def _row_source_id(row: dict[str, Any]) -> str:
    source_id = row.get("pair_source_id") or row.get("source_id")
    if source_id:
        return str(source_id)
    pair_id = str(row.get("pair_id") or "")
    return pair_id.split("__", 1)[0] if "__" in pair_id else pair_id


def _load_parsed(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed_path = row.get("parsed_output_path")
    if not parsed_path:
        return None
    p = Path(str(parsed_path))
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _text_from_row(row: dict[str, Any], field_name: str) -> str | None:
    parsed = _load_parsed(row)
    parsed_value = parsed.get(field_name) if parsed else None
    value = parsed_value if parsed_value not in {None, ""} else row.get(field_name)
    text = str(value or "").strip()
    return text or None


def _extract_recommendation(row: dict[str, Any]) -> str | None:
    text = _text_from_row(row, "recommendation")
    return text.lower() if text else None


def _extract_diagnosis_tokens(row: dict[str, Any]) -> set[str] | None:
    diag = _text_from_row(row, "diagnosis")
    if diag is None:
        return None
    tokens = {t for t in re.split(r"[^a-z0-9]+", diag.lower()) if len(t) > 2}
    return tokens


def _extract_severity_value(row: dict[str, Any]) -> float | None:
    sev = _text_from_row(row, "severity")
    if sev is None:
        return None
    sev = sev.lower()
    for key, val in STRUCTURED_SEVERITY_MAP_0_TO_3.items():
        if key in sev:
            return val
    return None


def _extract_confidence_value(row: dict[str, Any]) -> float | None:
    conf_text = _text_from_row(row, "confidence")
    if conf_text is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", conf_text)
    if m:
        return float(m.group(1))
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def compute_counterfactual_pair_metrics(rows: list[dict[str, Any]], tier_name: str) -> list[PairMetric]:
    by_model_pair_role: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("experiment_step") != "counterfactual":
            continue
        model = _row_model(row)
        pair_id = str(row.get("pair_id", ""))
        role = str(row.get("image_role", ""))
        if role not in {"source", "counterfactual"}:
            continue
        by_model_pair_role[(model, pair_id, role)] = row

    metrics: list[PairMetric] = []
    models_pairs: set[tuple[str, str]] = {(k[0], k[1]) for k in by_model_pair_role}
    for model, pair_id in sorted(models_pairs):
        src = by_model_pair_role.get((model, pair_id, "source"))
        cf = by_model_pair_role.get((model, pair_id, "counterfactual"))
        if not src or not cf:
            continue
        rec_src = _extract_recommendation(src)
        rec_cf = _extract_recommendation(cf)
        diag_src = _extract_diagnosis_tokens(src)
        diag_cf = _extract_diagnosis_tokens(cf)
        sev_src = _extract_severity_value(src)
        sev_cf = _extract_severity_value(cf)
        conf_src = _extract_confidence_value(src)
        conf_cf = _extract_confidence_value(cf)
        extracted_fields = (rec_src, rec_cf, diag_src, diag_cf, sev_src, sev_cf, conf_src, conf_cf)
        if any(value is None for value in extracted_fields):
            continue
        assert rec_src is not None and rec_cf is not None
        assert diag_src is not None and diag_cf is not None
        assert sev_src is not None and sev_cf is not None
        assert conf_src is not None and conf_cf is not None
        severity_disp = abs(sev_src - sev_cf)
        conf_disp = abs(conf_src - conf_cf)
        metrics.append(
            PairMetric(
                source_id=_row_source_id(src),
                pair_id=pair_id,
                model=model,
                tier=tier_name,
                recommendation_changed=float(rec_src != rec_cf),
                diagnostic_consistency=_jaccard(diag_src, diag_cf),
                severity_disparity=severity_disp,
                confidence_disparity=conf_disp,
            )
        )
    return metrics


def summarize_pair_metrics(metrics: list[PairMetric]) -> dict[str, dict[str, Any]]:
    _require_numeric_stack()
    by_model: dict[str, list[PairMetric]] = defaultdict(list)
    for m in metrics:
        by_model[m.model].append(m)

    out: dict[str, dict[str, Any]] = {}
    for model, vals in by_model.items():
        arr_rec = np.array([v.recommendation_changed for v in vals], dtype=float)
        arr_diag = np.array([v.diagnostic_consistency for v in vals], dtype=float)
        arr_sev = np.array([v.severity_disparity for v in vals], dtype=float)
        arr_conf = np.array([v.confidence_disparity for v in vals], dtype=float)

        out[model] = {
            "n_pairs": int(len(vals)),
            "recommendation_change_rate": float(arr_rec.mean()) if len(arr_rec) else 0.0,
            "diagnostic_consistency_mean": float(arr_diag.mean()) if len(arr_diag) else 0.0,
            "severity_disparity_mean": float(arr_sev.mean()) if len(arr_sev) else 0.0,
            "confidence_disparity_mean": float(arr_conf.mean()) if len(arr_conf) else 0.0,
        }
    return out


def bootstrap_ci(values: np.ndarray, n_iter: int = 10000, seed: int = 42) -> tuple[float, float]:
    _require_numeric_stack()
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_iter, len(values)), replace=True)
    stats_arr = np.mean(samples, axis=1)
    return float(np.quantile(stats_arr, 0.025)), float(np.quantile(stats_arr, 0.975))


def bootstrap_ci_source_clustered(
    values: np.ndarray,
    source_ids: np.ndarray,
    n_iter: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    if np is None:
        raise RuntimeError(
            "spinefairbench.analysis.endpoints requires optional analysis "
            "dependencies not installed on the reviewer path: numpy"
        )
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) != len(source_ids):
        raise ValueError(
            "source-clustered bootstrap requires equal-length values and source_ids"
        )
    from spinefairbench.release.scoring import source_clustered_bootstrap_ci

    return source_clustered_bootstrap_ci(
        [float(value) for value in values.tolist()],
        [str(source_id) for source_id in source_ids.tolist()],
        iterations=n_iter,
        seed=seed,
    )


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    _require_numeric_stack()
    if len(a) == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    if len(a) != len(b):
        raise ValueError(f"paired_wilcoxon requires equal-length inputs, got {len(a)} and {len(b)}")
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = stats.wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p)}


def bonferroni(p_values: list[float], alpha: float) -> list[dict[str, Any]]:
    n = max(len(p_values), 1)
    out: list[dict[str, Any]] = []
    for p in p_values:
        adj = min(1.0, p * n)
        out.append({"p_value": float(p), "adjusted_p": float(adj), "significant": bool(adj < alpha)})
    return out


def mitigation_effect_bounds(primary: float, mitigated: float) -> bool:
    return 0.0 <= primary <= 1.0 and 0.0 <= mitigated <= 1.0
