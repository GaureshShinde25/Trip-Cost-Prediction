"""
Trip Cost Predictor — Railway Fare Demonstration Script
========================================================
Demonstrates usage of RailwayFareService for calculating railway costs
across various destinations, classes, traveler counts, and origin options.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from railway.railway_fare_service import RailwayFareService


def print_result(example_title: str, res: dict):
    print("=" * 60)
    print(f" {example_title}")
    print("=" * 60)
    if res.get("status") == "SUCCESS":
        origin_str = (
            f"{res['origin_station']} ({res['origin_station_name']})"
            if res.get("origin_station")
            else "Not Specified (Network Aggregate)"
        )
        dest_str = f"{res['destination_station']} ({res['destination_station_name']})"
        train_str = (
            f"{res['train_number']} — {res['train_name']}"
            if res.get("train_number")
            else res.get("train_name", "N/A")
        )

        print(f"Destination:          {res['destination']}")
        print(f"Travel Class:         {res['travel_class']}")
        print(f"Travelers:            {res['travelers']}")
        print(f"Origin:               {origin_str}")
        print(f"Destination Station:  {dest_str}")
        print(f"Train:                {train_str}")
        print(f"Fare Per Person:      Rs. {res['fare_per_person']:,}")
        print(f"Total Railway Cost:   Rs. {res['total_railway_cost']:,}")
        print(f"Distance:             {res['distance_km']} km")
        print(f"Duration:             {res['duration_mins']} mins (~{res['duration_mins'] // 60}h {res['duration_mins'] % 60}m)")
        print(f"Fare Source:          {res['fare_source']}")
    else:
        print(f"Status:               ERROR")
        print(f"Error Message:        {res.get('error')}")
    print()


def main():
    service = RailwayFareService()
    print("\n" + "#" * 60)
    print(" TRIP COST PREDICTOR — RAILWAY FARE SERVICE DEMO")
    print("#" * 60 + "\n")

    # Example 1: Goa, 2 travelers, SL (Median Fallback)
    res1 = service.calculate_railway_cost(
        destination="Goa",
        travelers=2,
        travel_class="SL"
    )
    print_result("Example 1: Goa | 2 Travelers | Sleeper (SL)", res1)

    # Example 2: Mumbai, 3 travelers, 3A (Median Fallback)
    res2 = service.calculate_railway_cost(
        destination="Mumbai",
        travelers=3,
        travel_class="3A"
    )
    print_result("Example 2: Mumbai | 3 Travelers | AC 3 Tier (3A)", res2)

    # Example 3: Delhi, 2 travelers, 2A (Median Fallback)
    res3 = service.calculate_railway_cost(
        destination="Delhi",
        travelers=2,
        travel_class="2A"
    )
    print_result("Example 3: Delhi | 2 Travelers | AC 2 Tier (2A)", res3)

    # Example 4: Goa with Exact Origin Station (BSL -> Goa)
    res4 = service.calculate_railway_cost(
        destination="Goa",
        travelers=2,
        travel_class="SL",
        origin_station="BSL"
    )
    print_result("Example 4 (Bonus): Goa | 2 Travelers | SL | Origin = BSL (Exact)", res4)


if __name__ == "__main__":
    main()
