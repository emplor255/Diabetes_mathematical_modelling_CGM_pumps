#!/usr/bin/env python3
"""Regenerate dataset figures and counts from the supplied workbooks, without fitting."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_paper.data import load_and_prepare
from diabetes_paper.dataset_description import write_dataset_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hall", type=Path, default=ROOT / "hall2018_full_another_version.xlsx")
    parser.add_argument("--meals", type=Path, default=ROOT / "pbio.2005143.s015.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--min-event-points", type=int, default=24)
    args = parser.parse_args()
    prepared = load_and_prepare(args.hall, args.meals, min_event_points=args.min_event_points)
    write_dataset_artifacts(prepared, args.output)
    print(f"Dataset overview written to {args.output.resolve()}; no model fits were changed.")


if __name__ == "__main__":
    main()
