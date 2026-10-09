"""
Trip Cost Predictor — Railway Fare Service Unit Tests
=====================================================
Comprehensive tests validating RailwayFareService functionality,
exact fare retrieval, median estimation fallback, class normalization,
destination resolution, and input validation.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from railway.railway_fare_service import RailwayFareService


class TestRailwayFareService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Instantiate service once for all tests."""
        cls.service = RailwayFareService()

    def test_01_goa_sl_2_travelers(self):
        """TEST 1: Goa + SL + 2 travelers (Median fallback)."""
        res = self.service.calculate_railway_cost("Goa", travelers=2, travel_class="SL")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["destination"], "Goa")
        self.assertEqual(res["travel_class"], "SL")
        self.assertGreater(res["fare_per_person"], 0)
        self.assertEqual(res["total_railway_cost"], res["fare_per_person"] * 2)
        self.assertEqual(res["fare_source"], "DATASET_MEDIAN_ESTIMATE")
        self.assertIn(res["destination_station"], ["MAO", "KRMI", "THVM"])
        self.assertGreater(res["distance_km"], 0)
        self.assertGreater(res["duration_mins"], 0)

    def test_02_mumbai_3a_2_travelers(self):
        """TEST 2: Mumbai + 3A + 2 travelers (Median fallback)."""
        res = self.service.calculate_railway_cost("Mumbai", travelers=2, travel_class="3A")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["destination"], "Mumbai")
        self.assertEqual(res["travel_class"], "3A")
        self.assertGreater(res["fare_per_person"], 0)
        self.assertEqual(res["total_railway_cost"], res["fare_per_person"] * 2)
        self.assertEqual(res["fare_source"], "DATASET_MEDIAN_ESTIMATE")
        self.assertIn(res["destination_station"], ["MMCT", "CSMT", "DR", "BDTS"])

    def test_03_delhi_2a_3_travelers(self):
        """TEST 3: Delhi + 2A + 3 travelers (Median fallback)."""
        res = self.service.calculate_railway_cost("Delhi", travelers=3, travel_class="2A")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["destination"], "Delhi")
        self.assertEqual(res["travel_class"], "2A")
        self.assertGreater(res["fare_per_person"], 0)
        self.assertEqual(res["total_railway_cost"], res["fare_per_person"] * 3)
        self.assertEqual(res["fare_source"], "DATASET_MEDIAN_ESTIMATE")
        self.assertIn(res["destination_station"], ["NDLS", "DLI", "NZM", "ANVT"])

    def test_04_goa_sl_origin_station(self):
        """TEST 4: Goa + SL + origin station (BSL) -> EXACT_DATASET_FARE."""
        res = self.service.calculate_railway_cost(
            "Goa", travelers=2, travel_class="SL", origin_station="BSL"
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "EXACT_DATASET_FARE")
        self.assertEqual(res["origin_station"], "BSL")
        self.assertIn(res["destination_station"], ["MAO", "KRMI", "THVM"])
        self.assertGreater(res["fare_per_person"], 0)
        self.assertEqual(res["total_railway_cost"], res["fare_per_person"] * 2)
        self.assertIsNotNone(res["train_number"])
        self.assertIsNotNone(res["train_name"])

    def test_05_goa_invalid_class(self):
        """TEST 5: Goa + invalid class -> Returns clear validation error."""
        res = self.service.calculate_railway_cost("Goa", travelers=2, travel_class="INVALID_CLASS")
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("Unsupported travel class", res["error"])

    def test_06_invalid_destination(self):
        """TEST 6: Invalid destination -> Returns clear error."""
        res = self.service.calculate_railway_cost("Atlantis", travelers=2, travel_class="SL")
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("not found in destination mapping", res["error"])

    def test_07_no_origin_station_median_fallback(self):
        """TEST 7: No origin station -> Median fallback properly labeled."""
        res = self.service.calculate_railway_cost("Jaipur", travelers=4, travel_class="SL")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "DATASET_MEDIAN_ESTIMATE")
        self.assertIsNone(res["origin_station"])
        self.assertGreater(res["fare_per_person"], 0)
        self.assertEqual(res["total_railway_cost"], res["fare_per_person"] * 4)

    def test_08_train_specific_lookup(self):
        """TEST 8: Train-specific lookup using a valid train number (1139)."""
        res = self.service.calculate_railway_cost(
            "Goa", travelers=1, travel_class="SL", origin_station="BSL", train_number=1139
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["fare_source"], "EXACT_DATASET_FARE")
        self.assertEqual(res["train_number"], 1139)
        self.assertIn("NGP MAO SPECIAL", res["train_name"])

    def test_09_invalid_travelers(self):
        """TEST 9: Invalid travelers (negative / zero / non-int) -> Rejected."""
        res_zero = self.service.calculate_railway_cost("Goa", travelers=0, travel_class="SL")
        self.assertEqual(res_zero["status"], "ERROR")

        res_neg = self.service.calculate_railway_cost("Goa", travelers=-2, travel_class="SL")
        self.assertEqual(res_neg["status"], "ERROR")

        res_invalid = self.service.calculate_railway_cost("Goa", travelers="abc", travel_class="SL")
        self.assertEqual(res_invalid["status"], "ERROR")

    def test_10_all_35_destinations_resolve(self):
        """TEST 10: All 35 destinations resolve stations from mapping."""
        expected_35 = [
            "Agra", "Ahmedabad", "Alleppey", "Amritsar", "Bengaluru", "Chandigarh", "Chennai",
            "Coorg", "Darjeeling", "Delhi", "Gangtok", "Goa", "Guwahati", "Haridwar",
            "Hyderabad", "Jaipur", "Jodhpur", "Kochi", "Kolkata", "Lonavala", "Mahabaleshwar",
            "Manali", "Mumbai", "Munnar", "Mysuru", "Nashik", "Ooty", "Pondicherry", "Pune",
            "Rishikesh", "Shillong", "Shimla", "Udaipur", "Varanasi", "Wayanad"
        ]
        for dest in expected_35:
            stns = self.service.get_destination_stations(dest)
            self.assertGreater(len(stns), 0, f"Destination {dest} has no mapped stations!")
            # Also test case-insensitivity
            stns_lower = self.service.get_destination_stations(dest.lower())
            self.assertEqual(len(stns), len(stns_lower))

    def test_11_supported_classes(self):
        """TEST 11: All 6 standard classes are recognized."""
        for c in ["1A", "2A", "3A", "SL", "2S", "CC", "sl", "2a"]:
            norm = self.service.normalize_travel_class(c)
            self.assertIn(norm, {"1A", "2A", "3A", "SL", "2S", "CC"})

    def test_12_case_insensitivity(self):
        """TEST 12: Case insensitivity for destination and travel class."""
        res1 = self.service.calculate_railway_cost("goa", 2, "sl")
        res2 = self.service.calculate_railway_cost("GOA", 2, "SL")
        self.assertEqual(res1["fare_per_person"], res2["fare_per_person"])
        self.assertEqual(res1["travel_class"], "SL")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRailwayFareService)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
