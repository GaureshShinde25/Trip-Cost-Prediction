"""
Trip Cost Predictor — Gradio User Interface
===========================================
Interactive, user-friendly travel budget estimator supporting 35 Indian destinations,
6 travel modes, dynamic accommodation tiers, attraction previews, and dedicated
Indian Railways fare component calculation.

Safety Constraints:
- Zero external live API dependencies.
- ML prediction and railway fares are displayed as distinct, transparent components.
- Uses trained V2 pipeline without retraining.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import gradio as gr

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.prediction_service import TripCostPredictor
from ml.budget_breakdown import calculate_budget_breakdown
from railway.railway_fare_service import RailwayFareService

# File paths
DATA_DIR = PROJECT_ROOT / "data"
ATTRACTIONS_PATH = DATA_DIR / "destination_attractions_35.csv"
BUDGET_DATASET_PATH = DATA_DIR / "trip_budget_dataset_35_destinations.csv"
STATIONS_PATH = DATA_DIR / "railway_stations.csv"

# Global service instances (cached in memory)
predictor = TripCostPredictor()
railway_service = RailwayFareService()

# 1. Load destinations and attractions dynamically
df_attractions = pd.read_csv(ATTRACTIONS_PATH)
DESTINATIONS = sorted(df_attractions["Destination"].unique().tolist())
ATTRACTIONS_MAP: Dict[str, List[str]] = {}
for _, row in df_attractions.iterrows():
    dest = str(row["Destination"]).strip()
    places = [p.strip() for p in str(row["Places"]).split(",") if p.strip()]
    ATTRACTIONS_MAP[dest] = places

# 2. Load accommodation choices dynamically from training dataset
df_budget = pd.read_csv(BUDGET_DATASET_PATH)
ACCOMMODATION_TYPES = sorted(df_budget["Accommodation_Type"].unique().tolist())

# 3. Static travel modes and railway classes
TRAVEL_MODES = ["Bike", "Car", "Bus", "Train", "Flight", "Cruise"]
TRAVEL_CLASSES = ["SL", "3A", "2A", "1A", "CC", "2S"]

# 4. Load railway stations for origin selector
df_stations = pd.read_csv(STATIONS_PATH)
DEFAULT_ORIGIN = "None (Unspecified / Dataset Median Estimate)"
STATION_CHOICES = [DEFAULT_ORIGIN] + [
    f"{code} - {name}"
    for code, name in zip(df_stations["station_code"], df_stations["station_name"])
]


def format_inr(amount: float) -> str:
    """Formats a number using the Indian currency numbering convention."""
    val = int(round(amount))
    s = str(val)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    return "₹" + ",".join(parts) + "," + last_three


def format_duration(minutes: int) -> str:
    """Formats minutes into human-readable hours and minutes (e.g. 17h 10m)."""
    hrs = minutes // 60
    mins = minutes % 60
    if hrs > 0:
        return f"{hrs}h {mins:02d}m"
    return f"{mins}m"


def get_attractions_preview(destination: str, places_count: int = 3) -> str:
    """Builds markdown list of top attractions for a destination."""
    places = ATTRACTIONS_MAP.get(destination, [])
    if not places:
        return "No attractions information available."

    count = min(int(places_count), len(places))
    items = []
    for i, place in enumerate(places[:count], 1):
        items.append(f"**{i}.** {place}")

    remaining = len(places) - count
    if remaining > 0:
        items.append(f"\n*(+ {remaining} more available in {destination})*")

    return "\n".join(items)


def on_destination_change(destination: str) -> Tuple[Any, str]:
    """Callback when user selects a new destination."""
    places = ATTRACTIONS_MAP.get(destination, [])
    max_places = len(places) if places else 6
    default_val = min(3, max_places)
    preview_md = get_attractions_preview(destination, default_val)
    return gr.update(maximum=max_places, value=default_val), preview_md


def on_travel_mode_change(travel_mode: str) -> Any:
    """Callback to toggle train-specific options visibility."""
    is_train = travel_mode == "Train"
    return gr.update(visible=is_train)


def process_trip_prediction(
    destination: str,
    no_of_days: int,
    no_of_travelers: int,
    travel_mode: str,
    accommodation_type: str,
    places_count: int,
    travel_class: str,
    origin_station_choice: str,
) -> Tuple[str, str, Any, str, str]:
    """
    Main prediction coordinator:
    1. Validates trip parameters.
    2. Calls TripCostPredictor for total cost ML prediction.
    3. If Travel Mode == Train, calls RailwayFareService for railway itemization.
    4. Slices destination attractions.
    5. Returns formatted outputs for Gradio Blocks.
    """
    # ------------------ VALIDATION ------------------
    if not destination or destination not in ATTRACTIONS_MAP:
        err = f"⚠️ Please select a valid destination from the available list."
        return "", "", gr.update(visible=False), "", err

    try:
        days = int(no_of_days)
        if days <= 0:
            return "", "", gr.update(visible=False), "", "⚠️ Number of days must be greater than 0."
    except (ValueError, TypeError):
        return "", "", gr.update(visible=False), "", "⚠️ Please enter a valid number of days."

    try:
        travelers = int(no_of_travelers)
        if travelers <= 0:
            return "", "", gr.update(visible=False), "", "⚠️ Number of travelers must be greater than 0."
    except (ValueError, TypeError):
        return "", "", gr.update(visible=False), "", "⚠️ Please enter a valid number of travelers."

    available_places = ATTRACTIONS_MAP.get(destination, [])
    max_places = len(available_places)
    try:
        places = int(places_count)
        if places < 0:
            return "", "", gr.update(visible=False), "", "⚠️ Places count cannot be negative."
        if places > max_places:
            return (
                "",
                "",
                gr.update(visible=False),
                "",
                f"⚠️ Places count ({places}) exceeds maximum available attractions ({max_places}) for {destination}.",
            )
    except (ValueError, TypeError):
        return "", "", gr.update(visible=False), "", "⚠️ Please select a valid number of places."

    # ------------------ 1. ML TRIP COST PREDICTION ------------------
    ml_res = predictor.predict(
        destination=destination,
        no_of_days=days,
        no_of_travelers=travelers,
        travel_mode=travel_mode,
        accommodation_type=accommodation_type,
        places_count=places,
    )

    if ml_res.get("status") != "SUCCESS":
        err_msg = ml_res.get("error", "ML prediction failed.")
        return "", "", gr.update(visible=False), "", f"⚠️ {err_msg}"

    predicted_cost = ml_res["predicted_total_cost"]
    cost_formatted = format_inr(predicted_cost)

    # ------------------ 2. RAILWAY FARE COMPONENT (IF TRAIN) ------------------
    railway_md = ""
    railway_visible = False
    actual_rail_cost = None

    if travel_mode == "Train":
        railway_visible = True
        origin_code = None
        if origin_station_choice and not origin_station_choice.startswith("None"):
            origin_code = origin_station_choice.split(" - ")[0].strip()

        rail_res = railway_service.calculate_railway_cost(
            destination=destination,
            travelers=travelers,
            travel_class=travel_class,
            origin_station=origin_code,
        )

        if rail_res.get("status") == "SUCCESS":
            actual_rail_cost = rail_res["total_railway_cost"]
            fare_pp = format_inr(rail_res["fare_per_person"])
            tot_rail = format_inr(rail_res["total_railway_cost"])
            is_gw = rail_res.get("is_gateway", False)
            dest_display = f"{rail_res['destination_station_name']} ({rail_res['destination_station']})"
            dist_display = f"{rail_res['distance_km']} km"
            dur_display = format_duration(rail_res["duration_mins"])
            fare_src = rail_res["fare_source"]

            if fare_src == "EXACT_DATASET_FARE":
                origin_display = f"{rail_res['origin_station']} ({rail_res.get('origin_station_name', '')})"
                train_str = f"{rail_res.get('train_number')} - {rail_res.get('train_name', '')}"
                railway_md = f"""
### 🚆 Railway Fare Component

Fare Per Person:  
**{fare_pp}**  

Travelers:  
**{travelers}**  

Total Railway Fare:  
**<span style="color:#059669; font-weight:bold;">{tot_rail}</span>**  

Class:  
**{travel_class}**  

Origin:  
**{origin_display}**  

Destination:  
**{dest_display}**  

Train:  
**{train_str}**  

Distance:  
**{dist_display}**  

Duration:  
**{dur_display}**  

Source:  
`Exact Dataset Fare`  

| Parameter | Details |
| :--- | :--- |
| **Fare Per Person** | **{fare_pp}** |
| **Travelers** | **{travelers}** |
| **Total Railway Fare** | **<span style="color:#059669; font-weight:bold;">{tot_rail}</span>** |
| **Class** | **{travel_class}** |
| **Origin** | {origin_display} |
| **Destination** | {dest_display} |
| **Train** | {train_str} |
| **Distance** | {dist_display} |
| **Duration** | {dur_display} |
| **Source** | `Exact Dataset Fare` |
"""
            elif fare_src == "REVERSE_ROUTE_DATASET_ESTIMATE":
                origin_display = f"{rail_res['origin_station']} ({rail_res.get('origin_station_name', '')})"
                train_str = f"{rail_res.get('train_number')} - {rail_res.get('train_name', '')}"
                route_in_dataset = rail_res.get("route_in_dataset", "")
                railway_md = f"""
### 🚆 Railway Fare Component

Estimated Fare:  
**{fare_pp}/person**  

Travelers:  
**{travelers}**  

Estimated Railway Cost:  
**<span style="color:#059669; font-weight:bold;">{tot_rail}</span>**  

Source:  
`Reverse-Route Dataset Estimate`  

Train:  
**{train_str}**  

Route in Dataset:  
**{route_in_dataset}**  

Distance:  
**{dist_display}**  

Duration:  
**{dur_display}**  

| Parameter | Details |
| :--- | :--- |
| **Estimated Fare** | **{fare_pp}/person** |
| **Travelers** | **{travelers}** |
| **Estimated Railway Cost** | **<span style="color:#059669; font-weight:bold;">{tot_rail}</span>** |
| **Class** | **{travel_class}** |
| **Origin Station** | {origin_display} |
| **Destination** | {dest_display} |
| **Train in Dataset** | {train_str} |
| **Route in Dataset** | {route_in_dataset} |
| **Distance** | {dist_display} |
| **Duration** | {dur_display} |
| **Source** | `Reverse-Route Dataset Estimate` |

> ⚠️ **Warning**: *"Direct {rail_res.get('origin_station_name')} → {destination} fare was not available in the project dataset. This estimate uses an observed reverse-direction railway fare."*
"""
            elif is_gw and origin_code:
                # Gateway fallback for specified origin
                gw_display = f"{rail_res['gateway_station_name']} ({rail_res['gateway_station']})"
                origin_display = f"{rail_res['origin_station']} ({rail_res.get('origin_station_name', '')})"
                train_str = f"{rail_res.get('train_number')} - {rail_res.get('train_name', '')}"
                railway_md = f"""
### 🚆 Railway Fare Component

⚠️ **Direct route unavailable**

Gateway:  
**{gw_display}**  

Estimated Fare:  
**{fare_pp}/person**  

Total:  
**<span style="color:#059669; font-weight:bold;">{tot_rail}</span>**  

Source:  
`Gateway Dataset Estimate`  

| Parameter | Details |
| :--- | :--- |
| **Route Status** | ⚠️ Direct route unavailable |
| **Gateway** | **{gw_display}** |
| **Origin** | **{origin_display}** |
| **Estimated Fare** | **{fare_pp}/person** |
| **Travelers** | **{travelers}** |
| **Total Railway Fare** | **<span style="color:#059669; font-weight:bold;">{tot_rail}</span>** |
| **Class** | **{travel_class}** |
| **Train** | {train_str} |
| **Distance** | {dist_display} |
| **Duration** | {dur_display} |
| **Source** | `Gateway Dataset Estimate` |

> ℹ️ *Direct railway fare to {destination} was not found in the project dataset. A gateway station estimate is shown.*  
> ⚠️ **Disclaimer**: *"Gateway estimate does not include the onward road/local transfer from the gateway station to the destination."*
"""
            elif is_gw and not origin_code:
                # Gateway fallback for unspecified origin
                gw_display = f"{rail_res['gateway_station_name']} ({rail_res['gateway_station']})"
                railway_md = f"""
### 🚆 Railway Fare Component

⚠️ **Direct route unavailable**

Gateway:  
**{gw_display}**  

Estimated Fare:  
**{fare_pp}/person**  

Total:  
**<span style="color:#059669; font-weight:bold;">{tot_rail}</span>**  

Source:  
`Gateway Dataset Estimate`  

| Parameter | Details |
| :--- | :--- |
| **Route Status** | ⚠️ Direct route unavailable (Gateway Railhead) |
| **Gateway** | **{gw_display}** |
| **Origin** | *Not Specified (Dataset Median across network routes)* |
| **Estimated Fare** | **{fare_pp}/person** |
| **Travelers** | **{travelers}** |
| **Total Railway Fare** | **<span style="color:#059669; font-weight:bold;">{tot_rail}</span>** |
| **Class** | **{travel_class}** |
| **Distance (Median)** | {dist_display} |
| **Duration (Median)** | {dur_display} |
| **Source** | `Gateway Dataset Estimate` |

> ℹ️ *Direct railway connection to {destination} is unavailable; fares are estimated using railhead gateway {rail_res['gateway_station_name']}.*  
> ⚠️ **Disclaimer**: *"Gateway estimate does not include the onward road/local transfer from the gateway station to the destination."*
"""
            else:
                # Direct destination with median estimate
                railway_md = f"""
### 🚆 Railway Fare Component

Fare Per Person:  
**{fare_pp}**  

Travelers:  
**{travelers}**  

Total Railway Fare:  
**<span style="color:#059669; font-weight:bold;">{tot_rail}</span>**  

Class:  
**{travel_class}**  

Origin:  
*Not Specified (Dataset Median across network routes)*  

Destination:  
**{dest_display}**  

Distance:  
**{dist_display} (Median)**  

Duration:  
**{dur_display} (Median)**  

Source:  
`Dataset Median Estimate`  

| Parameter | Details |
| :--- | :--- |
| **Fare Per Person** | **{fare_pp}** |
| **Travelers** | **{travelers}** |
| **Total Railway Fare** | **<span style="color:#059669; font-weight:bold;">{tot_rail}</span>** |
| **Class** | **{travel_class}** |
| **Origin** | *Not Specified (Dataset Median across network routes)* |
| **Destination** | {dest_display} |
| **Distance (Median)** | {dist_display} |
| **Duration (Median)** | {dur_display} |
| **Source** | `Dataset Median Estimate` |
"""
        else:
            # Route not found error gracefully presented
            railway_md = f"""
### 🚆 Railway Fare Component

Status:  
`Fare Not Available`  

⚠️ No railway fare was found in the project dataset for this origin and destination.

Direct railway fare was not found for this origin and destination in the project dataset.

> **Next Steps**: Try another origin station or leave origin unspecified to use the dataset median estimate.
"""

    # ------------------ 3. BUDGET BREAKDOWN CALCULATION ------------------
    breakdown = calculate_budget_breakdown(
        total_cost=predicted_cost,
        no_of_days=days,
        no_of_travelers=travelers,
        travel_mode=travel_mode,
        accommodation_type=accommodation_type,
        places_count=places,
        actual_travel_cost=actual_rail_cost,
    )

    cost_pp = breakdown["cost_per_person"]
    cost_pp_formatted = format_inr(cost_pp)
    comp = breakdown["components"]
    shares = breakdown["shares"]

    travel_amt = format_inr(comp["Travel"])
    accom_amt = format_inr(comp["Accommodation"])
    food_amt = format_inr(comp["Food & Meals"])
    act_amt = format_inr(comp["Sightseeing & Activities"])
    local_amt = format_inr(comp["Local Transportation"])

    if breakdown.get("is_railway_fare"):
        fare_src = rail_res.get("fare_source", "")
        if fare_src == "EXACT_DATASET_FARE":
            travel_label = "🚆 Travel (Exact Dataset Fare)"
        elif fare_src == "REVERSE_ROUTE_DATASET_ESTIMATE":
            travel_label = "🚆 Travel (Reverse-Route Dataset Estimate)"
        elif fare_src == "GATEWAY_DATASET_ESTIMATE":
            travel_label = "🚆 Travel (Gateway Dataset Estimate)"
        elif fare_src == "DATASET_MEDIAN_ESTIMATE":
            travel_label = "🚆 Travel (Dataset Median Estimate)"
        else:
            travel_label = "🚆 Travel (Dataset Railway Fare)"
        disclaimer_text = breakdown["disclaimer"]
    else:
        if travel_mode == "Train":
            travel_label = "🚆 Travel (Estimated Allocation)"
            disclaimer_text = (
                "No railway fare was available in the project dataset. "
                "Budget breakdown is an estimated allocation of the predicted total and is not a live quotation."
            )
        else:
            mode_icons = {
                "Flight": "✈️",
                "Car": "🚗",
                "Bus": "🚌",
                "Bike": "🏍️",
                "Cruise": "🚢",
            }
            icon = mode_icons.get(travel_mode, "🚗")
            travel_label = f"{icon} Travel"
            disclaimer_text = breakdown["disclaimer"]

    prediction_md = f"""
### 💰 TOTAL TRIP BUDGET

# <span style="color:#1d4ed8; font-size:2.4rem; font-weight:800;">{cost_formatted}</span>

### 👤 {cost_pp_formatted} per person

*(Estimated Cost Per Person: {cost_pp_formatted} / person)*

---

### 📊 ESTIMATED BUDGET BREAKDOWN

{travel_label}  
**{travel_amt}** ({shares['Travel']}%)  

🏨 Accommodation  
**{accom_amt}** ({shares['Accommodation']}%)  

🍽️ Food & Meals  
**{food_amt}** ({shares['Food & Meals']}%)  

📍 Sightseeing & Activities  
**{act_amt}** ({shares['Sightseeing & Activities']}%)  

🚕 Local Transportation  
**{local_amt}** ({shares['Local Transportation']}%)  

────────────────────────  
### 💰 TOTAL  
**{cost_formatted}**  
────────────────────────  

| Category | Estimated Allocation | Share |
| :--- | :--- | :--- |
| {travel_label} | **{travel_amt}** | {shares['Travel']}% |
| 🏨 **Accommodation** | **{accom_amt}** | {shares['Accommodation']}% |
| 🍽️ **Food & Meals** | **{food_amt}** | {shares['Food & Meals']}% |
| 📍 **Sightseeing & Activities** | **{act_amt}** | {shares['Sightseeing & Activities']}% |
| 🚕 **Local Transportation** | **{local_amt}** | {shares['Local Transportation']}% |
| 💰 **Total Trip Cost** | **{cost_formatted}** | 100.0% |

> ℹ️ *{disclaimer_text}*
"""

    # ------------------ 3. PLACES TO VISIT & IMPORTANT INFO ------------------
    attractions_list = available_places[:places]
    places_items = []
    if attractions_list:
        for i, p in enumerate(attractions_list, 1):
            places_items.append(f"**{i}.** {p}")
        places_content = "\n".join(places_items)
    else:
        places_content = "*No specific sightseeing attractions selected in itinerary.*"

    places_md = f"""
### 📍 SELECTED PLACES TO VISIT

{places_content}

---

### ℹ️ IMPORTANT INFORMATION

• Total prediction is generated by the ML model.  
• Budget breakdown is an estimated allocation unless actual component data exists.  
• Railway fares come from the project's static railway dataset.  
• Railway prices are NOT live prices.  
• Gateway estimates do not include onward local transport unless explicitly calculated.  
"""

    return prediction_md, railway_md, gr.update(visible=railway_visible), places_md, ""


def create_ui():
    """Builds and returns the Gradio Blocks UI."""
    with gr.Blocks(title="Trip Cost Predictor") as demo:
        gr.Markdown(
            """
            # ✈️ TRIP COST PREDICTOR
            ### Smart Travel Budget & Railway Cost Planner
            """
        )

        with gr.Row():
            # ================= LEFT COLUMN: INPUTS =================
            with gr.Column(scale=1):
                gr.Markdown("#### 📋 1. Trip Configuration")

                dest_dropdown = gr.Dropdown(
                    choices=DESTINATIONS,
                    value="Goa",
                    label="Destination",
                    info="Choose from 35 supported Indian tourist destinations",
                    filterable=True,
                )

                with gr.Row():
                    days_input = gr.Number(
                        value=3,
                        label="Number of Days",
                        minimum=1,
                        precision=0,
                        info="Trip duration",
                    )
                    travelers_input = gr.Number(
                        value=2,
                        label="Number of Travelers",
                        minimum=1,
                        precision=0,
                        info="Passenger count",
                    )

                travel_mode_input = gr.Dropdown(
                    choices=TRAVEL_MODES,
                    value="Train",
                    label="Travel Mode",
                    info="Primary mode of transportation",
                )

                accommodation_input = gr.Dropdown(
                    choices=ACCOMMODATION_TYPES,
                    value="Budget Hotel",
                    label="Accommodation Type",
                    info="Preferred lodging standard",
                )

                places_slider = gr.Slider(
                    minimum=0,
                    maximum=6,
                    value=3,
                    step=1,
                    label="Number of Places to Visit",
                    info="Sightseeing attractions in itinerary",
                )

                # Train Options (Visible only when Travel Mode == Train)
                with gr.Group(visible=True) as train_options_group:
                    gr.Markdown("#### 🚆 Train Booking Options")
                    class_dropdown = gr.Dropdown(
                        choices=TRAVEL_CLASSES,
                        value="SL",
                        label="Travel Class",
                        info="Railway accommodation class",
                    )
                    origin_dropdown = gr.Dropdown(
                        choices=STATION_CHOICES,
                        value=DEFAULT_ORIGIN,
                        label="Origin Station (Optional)",
                        info="Leave unspecified for nationwide median fare estimate",
                        filterable=True,
                    )

                predict_btn = gr.Button(
                    "🔮 Predict Trip Cost",
                    variant="primary",
                    size="lg",
                )

                error_box = gr.Markdown(value="", label="Validation Notice")

            # ================= RIGHT COLUMN: OUTPUTS =================
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 2. Budget Estimation & Itinerary")

                prediction_box = gr.Markdown(
                    value="*Click 'Predict Trip Cost' to generate total budget estimation.*",
                    label="Trip Cost Prediction",
                )

                with gr.Group(visible=True) as railway_group:
                    railway_box = gr.Markdown(
                        value="*Railway breakdown will appear here when Travel Mode is Train.*",
                        label="Railway Fare Breakdown",
                    )

                places_box = gr.Markdown(
                    value=get_attractions_preview("Goa", 3),
                    label="Places to Visit",
                )

        # ================= DISCLAIMERS & MODEL INFO =================
        with gr.Accordion("ℹ️ Model Architecture & Technical Disclaimers", open=False):
            gr.Markdown(
                """
                - **Model**: Trip Cost Predictor V2 ([`HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html))
                - **Evaluation**: Mean Absolute Error (MAE) = **₹4,706.66**, $R^2$ Score = **0.9816** on 20% holdout test set.
                - **Training Dataset**: 3,500 synthetic trip records across 35 Indian destinations.
                - **Notice**: *Predictions are generated from the project's benchmark synthetic dataset and should not be interpreted as certified live market prices.*
                - **Railway Disclaimer**: *Railway fares are retrieved from the project's static Indian Railways snapshot (`railway_fares_clean.csv`) and do not reflect real-time IRCTC dynamic quotas.*
                """
            )

        # ================= DYNAMIC EVENT WIRING =================
        # 1. Travel mode change toggles railway options
        travel_mode_input.change(
            fn=on_travel_mode_change,
            inputs=[travel_mode_input],
            outputs=[train_options_group],
        )

        # 2. Destination change updates attractions preview and slider max
        dest_dropdown.change(
            fn=on_destination_change,
            inputs=[dest_dropdown],
            outputs=[places_slider, places_box],
        )

        # 3. Predict button executes end-to-end estimation
        predict_btn.click(
            fn=process_trip_prediction,
            inputs=[
                dest_dropdown,
                days_input,
                travelers_input,
                travel_mode_input,
                accommodation_input,
                places_slider,
                class_dropdown,
                origin_dropdown,
            ],
            outputs=[
                prediction_box,
                railway_box,
                railway_group,
                places_box,
                error_box,
            ],
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    # Launch strictly locally on 127.0.0.1 without public exposure
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
