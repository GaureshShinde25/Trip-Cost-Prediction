"""
Trip Cost Predictor — Railway Fare Resolution & Integrity Test Suite
=====================================================================
Comprehensive regression suite validating:
1. Priority 1: Exact Forward Dataset Fare (EXACT_DATASET_FARE)
2. Priority 2: Valid Gateway Dataset Fare (GATEWAY_DATASET_ESTIMATE)
3. Priority 3: Reverse-Route Dataset Estimate (REVERSE_ROUTE_DATASET_ESTIMATE)
4. Priority 4: Dataset Median Estimate (DATASET_MEDIAN_ESTIMATE)
5. Priority 5: No Fare Available graceful handling & friendly message
6. Chiplun (CHI) -> Mumbai railway fare resolution (DR -> CHI reverse: ₹245/person, ₹490 total)
7. Budget breakdown consistency: sum(components) == predicted_total
8. Zero double counting: railway fare is integrated into, not added to, ML predicted total
9. ML prediction stability: origin station does not alter ML total cost
10. Multi-route and multi-mode coverage (Goa, Ooty, Mumbai, Delhi, Car, Flight)
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


class TestRailwayFareResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rfs = RailwayFareService()
        cls.predictor = TripCostPredictor()
        cls.fares_df = pd.read_csv(PROJECT_ROOT / "data" / "railway_fares_clean.csv")
        cls.mapping_df = pd.read_csv(PROJECT_ROOT / "data" / "destination_railway_mapping.csv")

    # -------------------------------------------------------------------------
    # 1. Forward Exact Dataset Fare
    # -------------------------------------------------------------------------
    def test_01_forward_exact_fare(self):
        """Priority 1: Direct forward route returns EXACT_DATASET_FARE matching CSV."""
        res = self.rfs.calculate_railway_cost(
            destination="Goa",
            travelers=2,
            travel_class="SL",
            origin_station="CAN",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "EXACT_DATASET_FARE")
        self.assertFalse(res["is_gateway"])
        self.assertFalse(res["is_reverse"])
        self.assertEqual(res["origin_station"], "CAN")
        self.assertEqual(res["destination_station"], "MAO")

        # Verify fare matches raw CSV record
        forward_csv = self.fares_df[
            (self.fares_df["from_station_code"] == "CAN")
            & (self.fares_df["to_station_code"] == "MAO")
            & (self.fares_df["travel_class"] == "SL")
        ].sort_values(by=["distance_km", "total_fare", "train_number"])
        expected_fare = int(forward_csv.iloc[0]["total_fare"])
        self.assertEqual(res["fare_per_person"], expected_fare)
        self.assertEqual(res["total_railway_cost"], expected_fare * 2)

    # -------------------------------------------------------------------------
    # 2. Gateway Dataset Fare
    # -------------------------------------------------------------------------
    def test_02_gateway_fare(self):
        """Priority 2: Mapped gateway route returns GATEWAY_DATASET_ESTIMATE."""
        res = self.rfs.calculate_railway_cost(
            destination="Ooty",
            travelers=2,
            travel_class="SL",
            origin_station="MAS",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "GATEWAY_DATASET_ESTIMATE")
        self.assertTrue(res["is_gateway"])
        self.assertFalse(res["is_reverse"])
        self.assertEqual(res["destination_station"], "CBE")  # Coimbatore railhead
        self.assertIsNotNone(res["gateway_station"])
        self.assertIsNotNone(res["gateway_note"])

        # Verify fare matches raw CSV record for MAS -> CBE
        gw_csv = self.fares_df[
            (self.fares_df["from_station_code"] == "MAS")
            & (self.fares_df["to_station_code"] == "CBE")
            & (self.fares_df["travel_class"] == "SL")
        ].sort_values(by=["distance_km", "total_fare", "train_number"])
        expected_fare = int(gw_csv.iloc[0]["total_fare"])
        self.assertEqual(res["fare_per_person"], expected_fare)
        self.assertEqual(res["total_railway_cost"], expected_fare * 2)

    # -------------------------------------------------------------------------
    # 3. Reverse Estimate
    # -------------------------------------------------------------------------
    def test_03_reverse_estimate(self):
        """Priority 3: When forward is absent, uses reverse route marked REVERSE_ROUTE_DATASET_ESTIMATE."""
        res = self.rfs.calculate_railway_cost(
            destination="Mumbai",
            travelers=2,
            travel_class="SL",
            origin_station="CHI",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "REVERSE_ROUTE_DATASET_ESTIMATE")
        self.assertTrue(res["is_reverse"])
        self.assertEqual(res["origin_station"], "CHI")
        self.assertEqual(res["destination_station"], "DR")  # Dadar
        self.assertIn("DADAR", res["route_in_dataset"])
        self.assertIn("CHIPLUN", res["route_in_dataset"])
        self.assertIn("reverse-direction railway fare", res["reverse_warning"])

    # -------------------------------------------------------------------------
    # 4. Median Fallback
    # -------------------------------------------------------------------------
    def test_04_median_fallback(self):
        """Priority 4: When origin is omitted, calculates DATASET_MEDIAN_ESTIMATE across routes."""
        res = self.rfs.calculate_railway_cost(
            destination="Mumbai",
            travelers=2,
            travel_class="SL",
            origin_station=None,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "DATASET_MEDIAN_ESTIMATE")
        self.assertIsNone(res["origin_station"])
        self.assertGreater(res["fare_per_person"], 0)
        self.assertEqual(res["total_railway_cost"], res["fare_per_person"] * 2)

    # -------------------------------------------------------------------------
    # 5. No Fare Available Case
    # -------------------------------------------------------------------------
    def test_05_no_fare_case(self):
        """Priority 5: When neither forward, gateway, nor reverse exists, return graceful error."""
        res = self.rfs.calculate_railway_cost(
            destination="Ooty",
            travelers=2,
            travel_class="SL",
            origin_station="GDO",  # Gonda Jn has no route to CBE/Ooty
        )
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("No railway fare was found in the project dataset", res["error"])
        self.assertIn("Try another origin station or leave origin unspecified", res["error"])

        # Also verify UI process_trip_prediction gracefully handles this without crashing
        pred_md, rail_md, rail_grp, _, err = process_trip_prediction(
            destination="Ooty",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="GDO - GONDA JN",
        )
        self.assertEqual(err, "")
        self.assertIn("Fare Not Available", rail_md)
        self.assertIn("🚆 Travel (Estimated Allocation)", pred_md)
        self.assertIn("No railway fare was available in the project dataset", pred_md)

    # -------------------------------------------------------------------------
    # 6. Chiplun -> Mumbai Specific Verification
    # -------------------------------------------------------------------------
    def test_06_chiplun_to_mumbai_resolution(self):
        """Verify Chiplun -> Mumbai returns DR -> CHI reverse fare dynamically from CSV."""
        # Dynamically query raw dataset for reverse fare
        reverse_raw = self.fares_df[
            (self.fares_df["from_station_code"] == "DR")
            & (self.fares_df["to_station_code"] == "CHI")
            & (self.fares_df["travel_class"] == "SL")
        ]
        self.assertFalse(reverse_raw.empty, "Dataset must contain reverse route DR -> CHI for SL")
        expected_fare = int(reverse_raw.iloc[0]["total_fare"])
        self.assertEqual(expected_fare, 245, "Verified reverse dataset fare is ₹245")

        res = self.rfs.calculate_railway_cost(
            destination="Mumbai",
            travelers=2,
            travel_class="SL",
            origin_station="CHI",
        )
        self.assertEqual(res["fare_per_person"], expected_fare)
        self.assertEqual(res["total_railway_cost"], expected_fare * 2)
        self.assertEqual(res["total_railway_cost"], 490)
        self.assertEqual(res["train_number"], 22629)
        self.assertEqual(res["fare_source"], "REVERSE_ROUTE_DATASET_ESTIMATE")

        # Test end-to-end through UI prediction coordinator
        pred_md, rail_md, _, _, err = process_trip_prediction(
            destination="Mumbai",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="CHI - CHIPLUN",
        )
        self.assertEqual(err, "")
        self.assertIn("₹245/person", rail_md)
        self.assertIn("₹490", rail_md)
        self.assertIn("Reverse-Route Dataset Estimate", rail_md)
        self.assertIn("22629 - DADAR TEN EXP", rail_md)
        self.assertIn("DADAR (DR) → CHIPLUN (CHI)", rail_md)
        self.assertIn("🚆 Travel (Reverse-Route Dataset Estimate)", pred_md)
        self.assertIn("₹490", pred_md)

    # -------------------------------------------------------------------------
    # 7. Budget Breakdown Consistency
    # -------------------------------------------------------------------------
    def test_07_budget_breakdown_consistency(self):
        """Verify components sum exactly to predicted_total across diverse combinations."""
        test_cases = [
            # (dest, days, travelers, mode, accom, places, class, origin)
            ("Mumbai", 3, 2, "Train", "Budget Hotel", 3, "SL", "CHI - CHIPLUN"),
            ("Goa", 4, 2, "Train", "Budget Hotel", 3, "SL", "CAN - KANNUR"),
            ("Ooty", 3, 2, "Train", "Budget Hotel", 3, "SL", "MAS - MGR CHENNAI CTL"),
            ("Mumbai", 3, 2, "Train", "Standard Hotel", 2, "SL", "None"),
            ("Ooty", 3, 2, "Train", "Budget Hotel", 3, "SL", "GDO - GONDA JN"),
            ("Mumbai", 2, 2, "Car", "Standard Hotel", 2, "SL", "None"),
            ("Delhi", 4, 3, "Flight", "Luxury Hotel", 4, "SL", "None"),
        ]

        for dest, days, trav, mode, accom, pl, tclass, orig in test_cases:
            pred_res = self.predictor.predict(dest, days, trav, mode, accom, pl)
            total = pred_res["predicted_total_cost"]

            # Obtain railway cost if applicable
            rail_cost = None
            if mode == "Train":
                orig_code = orig.split(" - ")[0].strip() if orig and not orig.startswith("None") else None
                r = self.rfs.calculate_railway_cost(dest, trav, tclass, origin_station=orig_code)
                if r.get("status") == "SUCCESS":
                    rail_cost = r["total_railway_cost"]

            b = calculate_budget_breakdown(
                total_cost=total,
                no_of_days=days,
                no_of_travelers=trav,
                travel_mode=mode,
                accommodation_type=accom,
                places_count=pl,
                actual_travel_cost=rail_cost,
            )

            comp_sum = sum(b["components"].values())
            self.assertEqual(
                comp_sum,
                total,
                f"Discrepancy in breakdown sum for {dest} ({mode}): {comp_sum} != {total}",
            )

    # -------------------------------------------------------------------------
    # 8. Zero Double Counting
    # -------------------------------------------------------------------------
    def test_08_no_double_counting(self):
        """Verify railway fare is carved out of the total, NOT added on top."""
        pred_res = self.predictor.predict("Mumbai", 3, 2, "Train", "Budget Hotel", 3)
        total_ml = pred_res["predicted_total_cost"]

        # Chiplun railway cost is 490
        chi_rail_cost = 490
        b = calculate_budget_breakdown(
            total_cost=total_ml,
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            actual_travel_cost=chi_rail_cost,
        )

        self.assertEqual(b["components"]["Travel"], 490)
        # Sum must remain exactly total_ml, NOT total_ml + 490
        self.assertEqual(sum(b["components"].values()), total_ml)
        self.assertNotEqual(sum(b["components"].values()), total_ml + chi_rail_cost)
        # Remaining categories must sum to total_ml - 490
        non_travel_sum = (
            b["components"]["Accommodation"]
            + b["components"]["Food & Meals"]
            + b["components"]["Sightseeing & Activities"]
            + b["components"]["Local Transportation"]
        )
        self.assertEqual(non_travel_sum, total_ml - chi_rail_cost)

    # -------------------------------------------------------------------------
    # 9. ML Prediction Unchanged by Railway Origin
    # -------------------------------------------------------------------------
    def test_09_ml_prediction_unchanged(self):
        """ML total prediction remains invariant across railway origins."""
        baseline = self.predictor.predict("Mumbai", 3, 2, "Train", "Budget Hotel", 3)
        expected_total = baseline["predicted_total_cost"]

        origins = [
            "CHI - CHIPLUN",
            "PUNE - PUNE JN",
            "CSMT - C SHIVAJI MAH T",
            "None (Unspecified / Dataset Median Estimate)",
        ]

        for orig in origins:
            pred_md, _, _, _, err = process_trip_prediction(
                destination="Mumbai",
                no_of_days=3,
                no_of_travelers=2,
                travel_mode="Train",
                accommodation_type="Budget Hotel",
                places_count=3,
                travel_class="SL",
                origin_station_choice=orig,
            )
            self.assertEqual(err, "")
            # Check formatted ML total is identical in UI
            self.assertIn(f"₹{expected_total:,}", pred_md)

    # -------------------------------------------------------------------------
    # 10. Multi-Route and Non-Train Mode Tests (Step 13)
    # -------------------------------------------------------------------------
    def test_10_multi_route_and_modes(self):
        """Test Goa + exact origin, Goa + another origin, Ooty gateway, Mumbai car, Delhi flight."""
        # 1. Goa + Train + CAN
        r_can = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="CAN")
        # 2. Goa + Train + UD
        r_ud = self.rfs.calculate_railway_cost("Goa", 2, "SL", origin_station="UD")
        self.assertNotEqual(r_can["fare_per_person"], r_ud["fare_per_person"])

        # 3. Non-train modes: Car
        _, rail_car, grp_car, _, err_car = process_trip_prediction(
            "Mumbai", 2, 2, "Car", "Standard Hotel", 2, "SL", "None"
        )
        self.assertEqual(err_car, "")
        self.assertEqual(rail_car, "")
        self.assertFalse(grp_car.get("visible", True))

        # 4. Non-train modes: Flight
        _, rail_fl, grp_fl, _, err_fl = process_trip_prediction(
            "Delhi", 4, 3, "Flight", "Luxury Hotel", 4, "SL", "None"
        )
        self.assertEqual(err_fl, "")
        self.assertEqual(rail_fl, "")
        self.assertFalse(grp_fl.get("visible", True))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRailwayFareResolution)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
