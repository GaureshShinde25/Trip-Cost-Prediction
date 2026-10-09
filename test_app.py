"""
Trip Cost Predictor — UI & Application Integration Tests
========================================================
Comprehensive validation suite covering:
1. Goa + Train + no origin
2. Goa + Train + exact origin (BSL)
3. Ooty + Train + origin with no direct route (GDO)
4. Ooty + Train + gateway route (MAS -> CBE)
5. Ooty + Train + no origin (CBE median)
6. Mumbai + Car
7. Delhi + Flight
8. Luxury Hotel
9. Budget Hotel
10. Places Count = 0
11. Places Count = maximum
12. Invalid inputs (days <= 0, travelers <= 0, bad dest, excess places)
+ Exact budget breakdown sum reconciliation assertions across all scenarios.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    process_trip_prediction,
    create_ui,
    on_destination_change,
    on_travel_mode_change,
    format_inr,
    ATTRACTIONS_MAP,
    DESTINATIONS,
    predictor,
)
from ml.budget_breakdown import calculate_budget_breakdown
from railway.railway_fare_service import RailwayFareService


class TestTripCostPredictorRefinements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.railway_svc = RailwayFareService()

    # -------------------------------------------------------------
    # 1. Goa + Train + no origin
    # -------------------------------------------------------------
    def test_01_goa_train_no_origin(self):
        """1. Goa + Train + no origin -> Median estimate & budget breakdown."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="None (Unspecified / Dataset Median Estimate)",
        )
        self.assertEqual(err, "")
        self.assertIn("TOTAL TRIP BUDGET", pred_md)
        self.assertIn("ESTIMATED BUDGET BREAKDOWN", pred_md)
        self.assertIn("per person", pred_md)
        self.assertIn("Railway Fare Component", rail_md)
        self.assertIn("Dataset Median Estimate", rail_md)
        self.assertIn("IMPORTANT INFORMATION", places_md)

        # Verify budget breakdown reconciliation
        pred_res = predictor.predict("Goa", 3, 2, "Train", "Budget Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)
        self.assertEqual(b["cost_per_person"], int(round(total / 2)))

    # -------------------------------------------------------------
    # 2. Goa + Train + exact origin
    # -------------------------------------------------------------
    def test_02_goa_train_exact_origin(self):
        """2. Goa + Train + exact origin (BSL) -> Exact dataset fare."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="BSL - BHUSAVAL JN",
        )
        self.assertEqual(err, "")
        self.assertIn("Exact Dataset Fare", rail_md)
        self.assertIn("BSL", rail_md)
        self.assertIn("THVM", rail_md)

        pred_res = predictor.predict("Goa", 3, 2, "Train", "Budget Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)

    # -------------------------------------------------------------
    # 3. Ooty + Train + origin with no direct route
    # -------------------------------------------------------------
    def test_03_ooty_train_no_direct_route(self):
        """3. Ooty + Train + origin with no direct or gateway route (GDO) -> Friendly fallback, no crash."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Ooty",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="GDO - GONDA JN",
        )
        # Should NOT crash the application; ML prediction still rendered
        self.assertEqual(err, "")
        self.assertIn("TOTAL TRIP BUDGET", pred_md)
        self.assertIn("ESTIMATED BUDGET BREAKDOWN", pred_md)
        self.assertIn("Direct railway fare was not found", rail_md)
        self.assertIn("Try another origin station or leave origin unspecified", rail_md)

        pred_res = predictor.predict("Ooty", 3, 2, "Train", "Budget Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)

    # -------------------------------------------------------------
    # 4. Ooty + Train + gateway route if available
    # -------------------------------------------------------------
    def test_04_ooty_train_gateway_route(self):
        """4. Ooty + Train + gateway route (MAS -> Coimbatore CBE) -> Gateway Dataset Estimate."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Ooty",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="MAS - MGR CHENNAI CTL",
        )
        self.assertEqual(err, "")
        self.assertIn("Direct route unavailable", rail_md)
        self.assertIn("Gateway Dataset Estimate", rail_md)
        self.assertIn("COIMBATORE JN", rail_md)
        self.assertIn("Gateway estimate does not include the onward road/local transfer", rail_md)

        pred_res = predictor.predict("Ooty", 3, 2, "Train", "Budget Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)

    # -------------------------------------------------------------
    # 5. Ooty + Train + no origin
    # -------------------------------------------------------------
    def test_05_ooty_train_no_origin(self):
        """5. Ooty + Train + no origin -> Median Gateway Dataset Estimate."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Ooty",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
            travel_class="SL",
            origin_station_choice="None (Unspecified / Dataset Median Estimate)",
        )
        self.assertEqual(err, "")
        self.assertIn("Direct route unavailable", rail_md)
        self.assertIn("Gateway Dataset Estimate", rail_md)
        self.assertIn("COIMBATORE JN", rail_md)
        self.assertIn("Gateway estimate does not include the onward road/local transfer", rail_md)

        pred_res = predictor.predict("Ooty", 3, 2, "Train", "Budget Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)

    # -------------------------------------------------------------
    # 6. Mumbai + Car
    # -------------------------------------------------------------
    def test_06_mumbai_car(self):
        """6. Mumbai + Car -> Railway hidden, allocation reflects Car mode."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Mumbai",
            no_of_days=2,
            no_of_travelers=2,
            travel_mode="Car",
            accommodation_type="Standard Hotel",
            places_count=2,
            travel_class="SL",
            origin_station_choice="None",
        )
        self.assertEqual(err, "")
        self.assertEqual(rail_md, "")  # Railway component not populated for Car
        self.assertFalse(rail_grp.get("visible", True))

        pred_res = predictor.predict("Mumbai", 2, 2, "Car", "Standard Hotel", 2)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 2, 2, "Car", "Standard Hotel", 2)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)

    # -------------------------------------------------------------
    # 7. Delhi + Flight
    # -------------------------------------------------------------
    def test_07_delhi_flight(self):
        """7. Delhi + Flight -> Higher travel share than Bike."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Delhi",
            no_of_days=4,
            no_of_travelers=3,
            travel_mode="Flight",
            accommodation_type="Luxury Hotel",
            places_count=4,
            travel_class="SL",
            origin_station_choice="None",
        )
        self.assertEqual(err, "")
        self.assertEqual(rail_md, "")

        pred_res = predictor.predict("Delhi", 4, 3, "Flight", "Luxury Hotel", 4)
        total = pred_res["predicted_total_cost"]
        b_flight = calculate_budget_breakdown(total, 4, 3, "Flight", "Luxury Hotel", 4)
        b_bike = calculate_budget_breakdown(total, 4, 3, "Bike", "Luxury Hotel", 4)

        self.assertAlmostEqual(sum(b_flight["components"].values()), total, delta=1)
        # Flight travel share must be strictly higher than Bike travel share
        self.assertGreater(b_flight["components"]["Travel"], b_bike["components"]["Travel"])

    # -------------------------------------------------------------
    # 8. Luxury Hotel
    # -------------------------------------------------------------
    def test_08_luxury_hotel(self):
        """8. Luxury Hotel -> Higher accommodation allocation than Budget Hotel."""
        pred_res = predictor.predict("Jaipur", 3, 2, "Train", "Luxury Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b_lux = calculate_budget_breakdown(total, 3, 2, "Train", "Luxury Hotel", 3)
        b_bud = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)

        self.assertAlmostEqual(sum(b_lux["components"].values()), total, delta=1)
        self.assertGreater(b_lux["components"]["Accommodation"], b_bud["components"]["Accommodation"])

    # -------------------------------------------------------------
    # 9. Budget Hotel
    # -------------------------------------------------------------
    def test_09_budget_hotel(self):
        """9. Budget Hotel -> Higher accommodation allocation than Hostel."""
        pred_res = predictor.predict("Jaipur", 3, 2, "Train", "Budget Hotel", 3)
        total = pred_res["predicted_total_cost"]
        b_bud = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 3)
        b_hostel = calculate_budget_breakdown(total, 3, 2, "Train", "Hostel", 3)

        self.assertAlmostEqual(sum(b_bud["components"].values()), total, delta=1)
        self.assertGreater(b_bud["components"]["Accommodation"], b_hostel["components"]["Accommodation"])

    # -------------------------------------------------------------
    # 10. Places Count = 0
    # -------------------------------------------------------------
    def test_10_places_count_zero(self):
        """10. Places Count = 0 -> Activities share is minimized."""
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=0,
            travel_class="SL",
            origin_station_choice="None",
        )
        self.assertEqual(err, "")
        self.assertIn("No specific sightseeing attractions selected", places_md)

        pred_res = predictor.predict("Goa", 3, 2, "Train", "Budget Hotel", 0)
        total = pred_res["predicted_total_cost"]
        b_0 = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 0)
        b_6 = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", 6)

        self.assertAlmostEqual(sum(b_0["components"].values()), total, delta=1)
        self.assertLess(b_0["components"]["Sightseeing & Activities"], b_6["components"]["Sightseeing & Activities"])

    # -------------------------------------------------------------
    # 11. Places Count = maximum
    # -------------------------------------------------------------
    def test_11_places_count_maximum(self):
        """11. Places Count = maximum -> Full attraction list included & higher activities share."""
        max_places = len(ATTRACTIONS_MAP["Goa"])
        pred_md, rail_md, rail_grp, places_md, err = process_trip_prediction(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=max_places,
            travel_class="SL",
            origin_station_choice="None",
        )
        self.assertEqual(err, "")
        for place in ATTRACTIONS_MAP["Goa"]:
            self.assertIn(place, places_md)

        pred_res = predictor.predict("Goa", 3, 2, "Train", "Budget Hotel", max_places)
        total = pred_res["predicted_total_cost"]
        b = calculate_budget_breakdown(total, 3, 2, "Train", "Budget Hotel", max_places)
        self.assertAlmostEqual(sum(b["components"].values()), total, delta=1)

    # -------------------------------------------------------------
    # 12. Invalid input
    # -------------------------------------------------------------
    def test_12_invalid_input(self):
        """12. Invalid inputs return clear validation errors without crashing."""
        # Bad days
        _, _, _, _, err_days = process_trip_prediction("Goa", 0, 2, "Train", "Budget Hotel", 3, "SL", "None")
        self.assertIn("Number of days must be greater than 0", err_days)

        # Bad travelers
        _, _, _, _, err_trav = process_trip_prediction("Goa", 3, -1, "Train", "Budget Hotel", 3, "SL", "None")
        self.assertIn("Number of travelers must be greater than 0", err_trav)

        # Bad destination
        _, _, _, _, err_dest = process_trip_prediction("Atlantis", 3, 2, "Train", "Budget Hotel", 3, "SL", "None")
        self.assertIn("valid destination", err_dest)

        # Excess places
        _, _, _, _, err_places = process_trip_prediction("Goa", 3, 2, "Train", "Budget Hotel", 99, "SL", "None")
        self.assertIn("exceeds maximum available attractions", err_places)

    # -------------------------------------------------------------
    # Additional: UI & Callback checks
    # -------------------------------------------------------------
    def test_13_app_ui_initialization(self):
        """Verify Gradio Blocks UI initializes cleanly."""
        demo = create_ui()
        self.assertIsNotNone(demo)

    def test_14_dynamic_callbacks(self):
        """Verify dynamic interactive callbacks."""
        slider_up, prev_md = on_destination_change("Jaipur")
        self.assertIn("Hawa Mahal", prev_md)

        train_up = on_travel_mode_change("Train")
        self.assertTrue(train_up.get("visible", False))

        car_up = on_travel_mode_change("Car")
        self.assertFalse(car_up.get("visible", True))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTripCostPredictorRefinements)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
