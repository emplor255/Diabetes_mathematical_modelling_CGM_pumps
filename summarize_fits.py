#!/usr/bin/env python3
"""Build fit figures and tables from saved results, without refitting or touching controls."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_paper.fit_evaluation import fit_report_lines, write_fit_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    tables = args.output / "tables"
    evaluation = write_fit_artifacts(
        args.output, pd.read_csv(tables / "participant_parameters.csv"),
        pd.read_csv(tables / "model_predictions.csv"), pd.read_csv(tables / "model_comparison.csv"),
    )
    report = args.output / "analysis_report.md"
    if report.exists():
        content = report.read_text(encoding="utf-8")
        start = "### Cohort-wide fit evaluation\n"
        stop = "## Optimal control\n"
        if stop in content:
            before, after = content.split(stop, 1)
            if start in before:
                before = before.split(start, 1)[0]
            report.write_text(before.rstrip() + "\n\n" + "\n".join(fit_report_lines(evaluation))
                              + "\n" + stop + after, encoding="utf-8")
    print(f"Fit artifacts written to {args.output.resolve()}; saved fits and controls are unchanged.")


if __name__ == "__main__":
    main()
