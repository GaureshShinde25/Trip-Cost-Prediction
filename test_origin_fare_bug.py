"""
Regression Test Suite — Origin-Specific Railway Fare & Budget Breakdown
=========================================================================
Verifies:
1. Origin station is passed to RailwayFareService and uses exact from_station_code.
2. Exact fare lookup matches railway_fares_clean.csv records.
3. Different origins with different source fares produce different fares in both
   RailwayFareService and the Budget Breakdown Travel component.
4. Origins with genuinely identical dataset fares are preserved without artificial tampering.
5. ML predicted total cost remains unchanged across different railway origins.
6. Railway fare is not double-counted; budget breakdown components sum exactly to ML total.
7. Ooty + GDO (route not found in dataset) is handled gracefully without crashing.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from railway.railway_fare_service import RailwayFareService
from ml.prediction_service import TripCostPredictor
from ml.budget_breakdown import calculate_budget_breakdown
from app import process_trip_prediction


class TestOriginFareBug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rfs = RailwayFareService()
        cls.predictor = TripCostPredictor()
        cls.fares_df = pd.read_csv(PROJECT_ROOT / "data" / "railway_fares_clean.csv")

    # -------------------------------------------------------------------------
    # 1. An origin is actually passed to RailwayFareService
    # -------------------------------------------------------------------------
    def test_01_origin_passed_to_railway_fare_service(self):
        """Verify origin station parameter reaches service and sets from_station_code."""
        res_can = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="CAN")
        self.assertEqual(res_can["status"], "SUCCESS")
        self.assertEqual(res_can["origin_station"], "CAN")
        self.assertEqual(res_can["destination_station"], "MAO")

        res_ud = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="UD")
        self.assertEqual(res_ud["status"], "SUCCESS")
        self.assertEqual(res_ud["origin_station"], "UD")

    # -------------------------------------------------------------------------
    # 2. Exact fare lookup uses that origin and matches source data
    # -------------------------------------------------------------------------
    def test_02_exact_fare_matches_source_dataset(self):
        """Verify returned fare matches the raw clean dataset record for that origin."""
        res_can = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="CAN")
        can_raw = self.fares_df[
            (self.fares_df["from_station_code"] == "CAN")
            & (self.fares_df["to_station_code"] == "MAO")
            & (self.fares_df["travel_class"] == "SL")
        ].sort_values(by=["distance_km", "total_fare", "train_number"])

        expected_fare = int(can_raw.iloc[0]["total_fare"])
        self.assertEqual(res_can["fare_per_person"], expected_fare)
        self.assertEqual(res_can["total_railway_cost"], expected_fare * 2)
        self.assertEqual(res_can["fare_source"], "EXACT_DATASET_FARE")

    # -------------------------------------------------------------------------
    # 3. Different origins with different source fares return different fares
    # -------------------------------------------------------------------------
    def test_03_different_origins_produce_different_fares(self):
        """Verify distinct origins with different source fares return distinct fares."""
        # Test RailwayFareService directly for Goa (SL)
        res_can = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="CAN")
        res_ud = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="UD")
        res_qln = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="QLN")

        fare_can = res_can["fare_per_person"]
        fare_ud = res_ud["fare_per_person"]
        fare_qln = res_qln["fare_per_person"]

        self.assertNotEqual(fare_can, fare_ud, "CAN and UD fares should differ")
        self.assertNotEqual(fare_can, fare_qln, "CAN and QLN fares should differ")
        self.assertNotEqual(fare_ud, fare_qln, "UD and QLN fares should differ")

        # Test Ooty gateway routes: MAS vs SBC vs NZM
        res_mas = self.rfs.calculate_railway_cost("Ooty", 2, "SL", origin_station="MAS")
        res_sbc = self.rfs.calculate_railway_cost("Ooty", 2, "SL", origin_station="SBC")
        res_nzm = self.rfs.calculate_railway_cost("Ooty", 2, "SL", origin_station="NZM")

        self.assertNotEqual(res_mas["fare_per_person"], res_sbc["fare_per_person"])
        self.assertNotEqual(res_mas["fare_per_person"], res_nzm["fare_per_person"])

        # Test UI Budget Breakdown via process_trip_prediction
        p_can, _, _, _, _ = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 3, "SL", "CAN - KANNUR")
        p_ud, _, _, _, _ = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 3, "SL", "UD - UDUPI")
        p_qln, _, _, _, _ = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 3, "SL", "QLN - KOLLAM JN")

        # Extract Travel amounts from output
        self.assertIn("₹720", p_can, "CAN should display ₹720 travel cost in breakdown")
        self.assertIn("₹510", p_ud, "UD should display ₹510 travel cost in breakdown")
        self.assertIn("₹1,020", p_qln, "QLN should display ₹1,020 travel cost in breakdown")

    # -------------------------------------------------------------------------
    # 4. Origins with genuinely identical dataset fares are preserved
    # -------------------------------------------------------------------------
    def test_04_identical_dataset_fares_preserved(self):
        """If source data genuinely has the same fare for two origins, it is preserved."""
        # PGT (Palakkad) and ED (Erode) to CBE (Coimbatore) both cost ₹145 in dataset
        res_pgt = self.rfs.calculate_railway_cost("Ooty", 2, "SL", origin_station="PGT")
        res_ed = self.rfs.calculate_railway_cost("Ooty", 2, "SL", origin_station="ED")

        self.assertEqual(res_pgt["fare_per_person"], 145)
        self.assertEqual(res_ed["fare_per_person"], 145)
        self.assertEqual(res_pgt["fare_per_person"], res_ed["fare_per_person"])

    # -------------------------------------------------------------------------
    # 5. ML prediction remains unchanged by origin
    # -------------------------------------------------------------------------
    def test_05_ml_prediction_unchanged_by_origin(self):
        """ML total prediction does NOT vary by railway origin station."""
        ml_res = self.predictor.predict("Goa", 3, 2, "Train", "Budget Hotel", 3)
        expected_total = ml_res["predicted_total_cost"]

        p_can, _, _, _, _ = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 3, "SL", "CAN - KANNUR")
        p_ud, _, _, _, _ = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 3, "SL", "UD - UDUPI")
        p_none, _, _, _, _ = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 3, "SL", "None")

        formatted_total = f"₹{expected_total:,}"
        self.assertIn(formatted_total, p_can)
        self.assertIn(formatted_total, p_ud)
        self.assertIn(formatted_total, p_none)

    # -------------------------------------------------------------------------
    # 6. Railway fare is not double-counted; budget breakdown sums to ML total
    # -------------------------------------------------------------------------
    def test_06_budget_breakdown_reconciliation_exact(self):
        """Ensure component costs sum exactly to ML total cost with zero discrepancy."""
        total = 12869
        for rail_cost in [720, 510, 1020, 1300, 650, None]:
            b = calculate_budget_breakdown(
                total_cost=total,
                no_of_days=3,
                no_of_travelers=2,
                travel_mode="Train",
                accommodation_type="Budget Hotel",
                places_count=3,
                actual_travel_cost=rail_cost,
            )
            self.assertEqual(
                sum(b["components"].values()),
                total,
                f"Sum of breakdown components must equal {total}",
            )
            if rail_cost is not None:
                self.assertEqual(b["components"]["Travel"], rail_cost)

    # -------------------------------------------------------------------------
    # 7. Ooty + GDO (route not found in dataset) handled gracefully
    # -------------------------------------------------------------------------
    def test_07_ooty_gdo_handled_gracefully(self):
        """Ooty + GDO without direct/gateway route displays friendly message without crash."""
        pred_md, rail_md, _, _, err = process_trip_prediction(
            "Ooty", 3, 2, "Train", "Budget Hotel", 3, "SL", "GDO - GONDA JN"
        )
        self.assertEqual(err, "")
        self.assertIn("No railway fare was found in the project dataset for this origin and destination", rail_md)
        self.assertIn("TOTAL TRIP BUDGET", pred_md)
        self.assertIn("ESTIMATED BUDGET BREAKDOWN", pred_md)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOriginFareBug)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
