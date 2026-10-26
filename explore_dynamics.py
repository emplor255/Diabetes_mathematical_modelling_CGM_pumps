#!/usr/bin/env python3
"""Add ID-ordered fitting errors and parameter-sweep figures from a saved run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_paper.dynamics_exploration import SweepConfig, write_exploration_artifacts
from diabetes_paper.model import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reference-id", help="Default: participant nearest the cohort median RMSE")
    parser.add_argument("--horizon-hours", type=float, default=12)
    parser.add_argument("--step-minutes", type=float, default=1)
    parser.add_argument("--factors", nargs="+", type=float, default=[0.5, 0.75, 1, 1.25, 1.5])
    args = parser.parse_args()
    tables = args.output / "tables"
    with (tables / "data_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    model = ModelConfig(**audit["fit_config"]["model"])
    _, summary, metadata = write_exploration_artifacts(
        args.output, pd.read_csv(tables / "participant_parameters.csv"),
        pd.read_csv(tables / "model_predictions.csv"), pd.read_csv(tables / "model_comparison.csv"),
        reference_id=args.reference_id,
        sweep=SweepConfig(tuple(args.factors), args.horizon_hours, args.step_minutes), model=model,
    )
    print(f"New figures: {(args.output / 'figures').resolve()}")
    print(f"Reference: {metadata['reference_id']}; {len(summary)} scenarios; no refitting or control changes.")


if __name__ == "__main__":
    main()
