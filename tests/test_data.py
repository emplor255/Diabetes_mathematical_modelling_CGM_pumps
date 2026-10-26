import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from diabetes_paper.data import load_and_prepare
from diabetes_paper.dataset_description import (
    dataset_group_summary,
    meal_availability,
    meal_profile_sample_sizes,
    write_dataset_artifacts,
)


class WorkbookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.prepared = load_and_prepare(
            root / "hall2018_full_another_version.xlsx",
            root / "pbio.2005143.s015.xlsx",
        )

    def test_expected_cohort_counts(self):
        audit = self.prepared.audit
        self.assertEqual(audit["hall_rows"], 105417)
        self.assertEqual(audit["hall_participants"], 57)
        self.assertEqual(audit["hall_healthy_participants"], 38)
        self.assertEqual(audit["hall_prediabetes_participants"], 14)
        self.assertEqual(audit["hall_type2_participants"], 5)

    def test_meal_qc_is_reproducible(self):
        audit = self.prepared.audit
        self.assertEqual(audit["meal_events_total"], 176)
        self.assertEqual(audit["meal_events_usable"], 147)
        self.assertEqual(audit["meal_duplicate_rows_collapsed"], 87)
        self.assertEqual(audit["meal_censored_low_rows"], 6)

    def test_usable_windows_have_unambiguous_anchor(self):
        usable_ids = set(
            self.prepared.meal_qc.loc[self.prepared.meal_qc["usable"], "event_id"]
        )
        points = self.prepared.meal_points.loc[
            self.prepared.meal_points["event_id"].isin(usable_ids)
        ]
        ranges = points.groupby("event_id")["relative_min"].agg(["min", "max"])
        self.assertTrue(ranges["min"].eq(-30.0).all())
        self.assertTrue(ranges["max"].eq(150.0).all())

    def test_nominal_slots_distinguish_missing_and_excluded(self):
        slots = meal_availability(self.prepared)
        self.assertEqual(len(slots), 180)
        self.assertFalse(slots.duplicated(["userID", "Meal"]).any())
        self.assertEqual(slots.availability.value_counts().to_dict(),
                         {"Retained": 147, "Excluded": 29, "Absent": 4})
        absent = set(slots.loc[slots.availability.eq("Absent"), ["userID", "Meal"]]
                     .itertuples(index=False, name=None))
        self.assertEqual(absent, {
            ("2133-017", "PB 2"), ("2133-018", "PB 2"),
            ("2133-041", "Bar 1"), ("2133-041", "PB 2"),
        })

    def test_dataset_denominators_and_meal_counts(self):
        slots = meal_availability(self.prepared)
        summary = dataset_group_summary(self.prepared, slots).set_index("status")
        overall = summary.loc["All participants"]
        self.assertEqual(overall.hall_participants, 57)
        self.assertEqual(overall.meal_participants, 30)
        self.assertEqual(overall.retained_participants, 30)
        self.assertEqual(overall.hall_cgm_rows, 105417)
        self.assertEqual(overall.retained_windows, 147)
        self.assertEqual(list(overall[["retained_PB_windows", "retained_Bar_windows", "retained_CF_windows"]]),
                         [51, 47, 49])
        self.assertEqual(list(summary.loc[["Healthy", "Prediabetes", "Type 2 diabetes"], "retained_windows"]),
                         [101, 32, 14])
        self.assertAlmostEqual(overall.meal_age_mean, 41.9)
        self.assertTrue((summary.observed_windows == summary.retained_windows + summary.excluded_windows).all())

    def test_figure_two_counts_are_people_not_replicates(self):
        counts = meal_profile_sample_sizes(self.prepared.meal_points)
        healthy_pb = counts.loc[counts.status.eq("Healthy") & counts.meal_type.eq("PB")]
        self.assertEqual(healthy_pb.participants.max(), 20)
        self.assertEqual(healthy_pb.events.max(), 36)
        type2_pb = counts.loc[counts.status.eq("Type 2 diabetes") & counts.meal_type.eq("PB")]
        self.assertEqual(type2_pb.participants.min(), 2)
        self.assertEqual(type2_pb.participants.max(), 3)

    def test_dataset_exports_without_fitting(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_dataset_artifacts(self.prepared, output)
            for relative in [
                "figures/figure_6_dataset_overview.png", "figures/figure_6_dataset_overview.pdf",
                "tables/dataset_overview.csv", "tables/dataset_overview.tex",
                "tables/dataset_meal_availability.csv", "tables/meal_profile_sample_sizes.csv",
            ]:
                self.assertGreater((output / relative).stat().st_size, 0)
            audit = json.loads((output / "tables/dataset_description_audit.json").read_text())
            self.assertEqual(audit["retained_unique_points"], 5313)
            self.assertEqual(audit["retained_premeal_points"], 861)
            self.assertEqual(audit["retained_postmeal_points_including_zero"], 4452)
            self.assertEqual(audit["meal_min_event_points"], 24)
            exported = pd.read_csv(output / "tables/dataset_overview.csv")
            self.assertEqual(exported.iloc[0].retained_windows, 147)


if __name__ == "__main__":
    unittest.main()
