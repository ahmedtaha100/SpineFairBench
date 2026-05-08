from __future__ import annotations

"""Internal full-analysis entry point intentionally omitted from the release.

The public reviewer package reproduces frozen results through
``reviewer_verify.py`` and the released artifact summaries. The private
end-to-end analysis CLI depended on unreleased run orchestration, plotting,
statistics, and reporting modules, so this public module fails explicitly
instead of exposing stale or partially redacted analysis code.
"""

import argparse
from typing import Sequence

INTERNAL_ANALYSIS_MODULES = (
    "spinefairbench.analysis.report",
    "spinefairbench.analysis.statistics",
    "spinefairbench.analysis.tables",
    "spinefairbench.analysis.visualization",
    "spinefairbench.config.schemas",
    "spinefairbench.config.settings",
    "spinefairbench.data.annotations",
    "spinefairbench.metrics.aggregator",
    "spinefairbench.metrics.llm_judge",
    "spinefairbench.utils.logging",
)


def _internal_analysis_error() -> RuntimeError:
    return RuntimeError(
        "spinefairbench.analyze is an internal full-analysis CLI and is not "
        "included in the reviewer verification package. Use reviewer_verify.py "
        "with the released frozen artifacts. Omitted private modules: "
        f"{', '.join(INTERNAL_ANALYSIS_MODULES)}"
    )


def _require_full_analysis_runtime() -> None:
    raise _internal_analysis_error()


def run_analysis(args: argparse.Namespace) -> None:
    _require_full_analysis_runtime()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal SpineFairBench analysis CLI placeholder."
    )
    parser.add_argument("--config", help="Internal-only analysis config path.")
    parser.add_argument("--results-dir", help="Internal-only results directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    _require_full_analysis_runtime()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
