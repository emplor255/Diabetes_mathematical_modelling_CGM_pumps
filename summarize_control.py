#!/usr/bin/env python3
"""Add cohort control tables to an existing run, without refitting or reoptimizing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_paper.reporting import write_analysis_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    tables = args.output / "tables"
    with (tables / "data_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    participants = pd.read_csv(tables / "participant_parameters.csv")
    comparisons = pd.read_csv(tables / "model_comparison.csv")
    try:
        summary = pd.read_csv(tables / "control_summary.csv")
    except pd.errors.EmptyDataError:
        parser.error("This run has no control simulations; generate them before summarizing.")
    write_analysis_report(args.output / "analysis_report.md", audit, participants, comparisons, summary)
    print(f"Control tables for {len(participants)} participants: {tables.resolve()}")
    print(f"Updated report: {(args.output / 'analysis_report.md').resolve()}")


if __name__ == "__main__":
    main()
