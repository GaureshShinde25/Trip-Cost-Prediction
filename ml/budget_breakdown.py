"""
Trip Cost Predictor — Budget Breakdown Allocator
=================================================
Calculates a deterministic, documented, and reproducible allocation of the ML-predicted
Total_Cost into 5 core trip expense categories:
1. 🚆 Travel
2. 🏨 Accommodation
3. 🍽️ Food & Meals
4. 📍 Sightseeing & Activities
5. 🚕 Local Transportation

Critical Design Constraints:
- Sum of all component costs equals Total_Cost exactly (to ₹0 difference).
- Responds logically to user trip choices (Flight vs Bike, Luxury vs Hostel, places count, days).
- Transparently disclaims that values represent estimated allocations rather than live market quotations.
"""

from typing import Dict, Any, Tuple, Optional


def calculate_budget_breakdown(
    total_cost: int,
    no_of_days: int,
    no_of_travelers: int,
    travel_mode: str,
    accommodation_type: str,
    places_count: int,
    actual_travel_cost: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Allocates total_cost across travel, lodging, food, sightseeing, and local transport.
    If actual_travel_cost is provided (e.g. dataset railway fare for Train mode),
    allocates that exact amount to Travel and distributes remaining budget across
    the other categories. Guarantees exact summation to total_cost.
    """
    total = int(round(total_cost))
    days = max(1, int(no_of_days))
    travelers = max(1, int(no_of_travelers))
    places = max(0, int(places_count))

    # Base allocation weights
    w_travel = 25.0
    w_accom = 35.0
    w_food = 20.0
    w_activities = 10.0
    w_local = 10.0

    # 1. Travel mode modifiers
    mode_weights = {
        "Flight": 1.85,
        "Cruise": 1.50,
        "Car": 1.10,
        "Train": 1.00,
        "Bus": 0.80,
        "Bike": 0.55,
    }
    w_travel *= mode_weights.get(travel_mode, 1.0)

    # 2. Accommodation modifiers
    accom_weights = {
        "Resort": 1.70,
        "Luxury Hotel": 1.60,
        "Standard Hotel": 1.00,
        "Budget Hotel": 0.80,
        "Homestay": 0.70,
        "Hostel": 0.50,
    }
    w_accom *= accom_weights.get(accommodation_type, 1.0)

    # 3. Sightseeing & Activities modifier (scales with places count)
    # Range: 0.20 (0 places) to 1.60 (6 places)
    places_factor = 0.20 + (0.233 * min(6, places))
    w_activities *= places_factor

    # 4. Duration modifier (longer stay increases daily lodging/food share relative to travel)
    day_scale = 1.0 + 0.02 * (days - 3)
    w_accom *= max(0.75, day_scale)
    w_food *= max(0.75, day_scale)
    w_local *= max(0.75, 1.0 + 0.015 * (days - 3))

    # Mode: Train with known dataset railway cost
    if actual_travel_cost is not None and travel_mode == "Train" and 0 < actual_travel_cost < total:
        travel_cost = int(round(actual_travel_cost))
        remaining = total - travel_cost
        non_travel_w = w_accom + w_food + w_activities + w_local

        components = {
            "Travel": travel_cost,
            "Accommodation": int(round(remaining * (w_accom / non_travel_w))),
            "Food & Meals": int(round(remaining * (w_food / non_travel_w))),
            "Sightseeing & Activities": int(round(remaining * (w_activities / non_travel_w))),
            "Local Transportation": int(round(remaining * (w_local / non_travel_w))),
        }
        diff = total - sum(components.values())
        # Reconcile rounding difference on the largest non-travel category
        largest_key = max(
            ["Accommodation", "Food & Meals", "Sightseeing & Activities", "Local Transportation"],
            key=components.get,
        )
        components[largest_key] += diff
        shares = {k: round((v / total) * 100, 1) for k, v in components.items()}
        is_railway_fare = True
        disclaimer = (
            "Budget breakdown incorporates the railway fare from the project dataset, "
            "with remaining predicted budget allocated across other trip categories."
        )
    else:
        total_weight = w_travel + w_accom + w_food + w_activities + w_local
        shares_raw = {
            "Travel": w_travel / total_weight,
            "Accommodation": w_accom / total_weight,
            "Food & Meals": w_food / total_weight,
            "Sightseeing & Activities": w_activities / total_weight,
            "Local Transportation": w_local / total_weight,
        }
        components = {k: int(round(total * v)) for k, v in shares_raw.items()}
        diff = total - sum(components.values())
        largest_key = max(components, key=components.get)
        components[largest_key] += diff
        shares = {k: round(v * 100, 1) for k, v in shares_raw.items()}
        is_railway_fare = False
        disclaimer = (
            "Budget breakdown is an estimated allocation of the predicted total "
            "and is not a live quotation."
        )

    cost_per_person = int(round(total / travelers))

    return {
        "total_cost": total,
        "cost_per_person": cost_per_person,
        "travelers": travelers,
        "components": components,
        "shares": shares,
        "is_estimated": not is_railway_fare,
        "is_railway_fare": is_railway_fare,
        "disclaimer": disclaimer,
    }
