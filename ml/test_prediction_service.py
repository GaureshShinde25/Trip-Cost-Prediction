"""
Trip Cost Predictor — Prediction Service Unit Tests
===================================================
Tests functionality, input validation, and numerical stability of TripCostPredictor.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.prediction_service import TripCostPredictor


class TestTripCostPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize prediction service once."""
        cls.predictor = TripCostPredictor()

    def test_01_goa_train(self):
        """TEST 1: Goa, 3 days, 2 travelers, Train, Budget Hotel, 3 places."""
        res = self.predictor.predict(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=3,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("predicted_total_cost", res)
        self.assertIsInstance(res["predicted_total_cost"], int)
        self.assertGreater(res["predicted_total_cost"], 0)
        self.assertEqual(res["currency"], "INR")
        self.assertEqual(res["model_version"], "v2")

    def test_02_mumbai_car(self):
        """TEST 2: Mumbai, 2 days, 2 travelers, Car, Standard Hotel, 2 places."""
        res = self.predictor.predict(
            destination="Mumbai",
            no_of_days=2,
            no_of_travelers=2,
            travel_mode="Car",
            accommodation_type="Standard Hotel",
            places_count=2,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["predicted_total_cost"], 0)

    def test_03_delhi_flight(self):
        """TEST 3: Delhi, 4 days, 3 travelers, Flight, Luxury Hotel, 4 places."""
        res = self.predictor.predict(
            destination="Delhi",
            no_of_days=4,
            no_of_travelers=3,
            travel_mode="Flight",
            accommodation_type="Luxury Hotel",
            places_count=4,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["predicted_total_cost"], 0)

    def test_04_manali_bus(self):
        """TEST 4: Manali, 5 days, 4 travelers, Bus, Homestay, 3 places."""
        res = self.predictor.predict(
            destination="Manali",
            no_of_days=5,
            no_of_travelers=4,
            travel_mode="Bus",
            accommodation_type="Homestay",
            places_count=3,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["predicted_total_cost"], 0)

    def test_05_unseen_destination_graceful_handling(self):
        """TEST 5: Unseen destination (OneHotEncoder handles unknown without crashing)."""
        res = self.predictor.predict(
            destination="Atlantis",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Flight",
            accommodation_type="Resort",
            places_count=2,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["predicted_total_cost"], 0)

    def test_06_zero_days_rejected(self):
        """TEST 6: Zero days returns clear validation error."""
        res = self.predictor.predict(
            destination="Goa",
            no_of_days=0,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=2,
        )
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("Invalid no_of_days", res["error"])

    def test_07_negative_travelers_rejected(self):
        """TEST 7: Negative travelers returns clear validation error."""
        res = self.predictor.predict(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=-2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=2,
        )
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("Invalid no_of_travelers", res["error"])

    def test_08_zero_places_accepted(self):
        """TEST 8: Zero places count is valid and accepted."""
        res = self.predictor.predict(
            destination="Goa",
            no_of_days=3,
            no_of_travelers=2,
            travel_mode="Train",
            accommodation_type="Budget Hotel",
            places_count=0,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["predicted_total_cost"], 0)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTripCostPredictor)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
