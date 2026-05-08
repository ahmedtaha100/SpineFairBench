from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional analysis dependency
    matplotlib = None
    plt = None

MODEL_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

DEMOGRAPHIC_ORDER = [
    "elderly_female",
    "elderly_male",
    "young_female",
    "young_male",
    "no_demographic",
]


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "spinefairbench.analysis.figure_style requires matplotlib, which "
            "is not installed on the reviewer verification path"
        )


def apply_style() -> None:
    _require_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 200,
        }
    )


def caption_text(
    *,
    subset: str,
    tier: str,
    experiment: str,
    n_pairs: int,
    n_sources: int,
    test_name: str,
    correction: str,
    claim_class: str,
) -> str:
    return (
        f"subset={subset}; tier={tier}; experiment={experiment}; n_pairs={n_pairs}; "
        f"n_sources={n_sources}; test={test_name}; correction={correction}; {claim_class}"
    )


def export_pending_figure(base_path: Path, title: str, note: str) -> list[Path]:
    _require_matplotlib()
    apply_style()
    outputs: list[Path] = []
    for width_name, size in (("single", (6, 4)), ("double", (12, 4))):
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(0.5, 0.42, "PENDING", ha="center", va="center", fontsize=11, color="darkred")
        ax.text(0.5, 0.25, note, ha="center", va="center", fontsize=8, wrap=True)

        png_path = base_path.with_name(f"{base_path.stem}.{width_name}.png")
        pdf_path = base_path.with_name(f"{base_path.stem}.{width_name}.pdf")
        csv_path = base_path.with_name(f"{base_path.stem}.{width_name}.csv")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["status", "note"])
            writer.writerow(["pending", note])
        outputs.extend([png_path, pdf_path, csv_path])
    return outputs


def _escape_tex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def write_pending_table(path_no_ext: Path, columns: list[str], row: list[Any]) -> list[Path]:
    if len(columns) != len(row):
        raise ValueError(f"columns and row must have the same length (got {len(columns)} and {len(row)})")
    csv_path = path_no_ext.with_suffix(".csv")
    tex_path = path_no_ext.with_suffix(".tex")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerow([str(v) for v in row])
    tex = "\\begin{tabular}{" + "c" * len(columns) + "}\n"
    tex += " & ".join(_escape_tex(column) for column in columns) + " \\\\ \n"
    tex += " & ".join(_escape_tex(value) for value in row) + " \\\\ \n"
    tex += "\\end{tabular}\n"
    tex_path.write_text(tex, encoding="utf-8")
    return [csv_path, tex_path]
