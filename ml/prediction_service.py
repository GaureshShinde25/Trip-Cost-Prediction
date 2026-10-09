"""
Trip Cost Predictor — ML Prediction Service
============================================
Provides clean, validated trip cost inference utilizing models/trip_budget_model_v2.pkl.
Does NOT retrain the model; loads the persisted pipeline once upon instantiation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import format_input_record

MODEL_V2_PATH = PROJECT_ROOT / "models" / "trip_budget_model_v2.pkl"


class TripCostPredictor:
    """
    Inference service for predicting total trip cost.
    Loads and caches trip_budget_model_v2.pkl.
    """

    VALID_MODES = {"Bike", "Car", "Bus", "Train", "Flight", "Cruise"}
    VALID_ACCOMMODATIONS = {
        "Homestay",
        "Hostel",
        "Budget Hotel",
        "Standard Hotel",
        "Luxury Hotel",
        "Resort",
    }

    def __init__(self, model_path: Union[str, Path] = MODEL_V2_PATH):
        self.model_path = Path(model_path)
        self._load_model()

    def _load_model(self) -> None:
        """Loads persisted pipeline without retraining."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please execute train_model.py first."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(
        self,
        destination: str,
        no_of_days: int,
        no_of_travelers: int,
        travel_mode: str,
        accommodation_type: str,
        places_count: int,
    ) -> Dict[str, Any]:
        """
        Predicts total trip budget from trip attributes.

        Validations:
        - no_of_days > 0
        - no_of_travelers > 0
        - places_count >= 0
        - non-empty categorical values

        Returns:
        {
            "predicted_total_cost": int,
            "currency": "INR",
            "model_version": "v2",
            "status": "SUCCESS"
        }
        """
        # Validate days
        try:
            days = int(no_of_days)
            if days <= 0:
                return {
                    "status": "ERROR",
                    "error": f"Invalid no_of_days: {no_of_days}. Days must be an integer > 0.",
                }
        except (ValueError, TypeError):
            return {
                "status": "ERROR",
                "error": f"Invalid no_of_days: '{no_of_days}'. Must be an integer.",
            }

        # Validate travelers
        try:
            travelers = int(no_of_travelers)
            if travelers <= 0:
                return {
                    "status": "ERROR",
                    "error": f"Invalid no_of_travelers: {no_of_travelers}. Travelers must be an integer > 0.",
                }
        except (ValueError, TypeError):
            return {
                "status": "ERROR",
                "error": f"Invalid no_of_travelers: '{no_of_travelers}'. Must be an integer.",
            }

        # Validate places count
        try:
            places = int(places_count)
            if places < 0:
                return {
                    "status": "ERROR",
                    "error": f"Invalid places_count: {places_count}. Places count must be >= 0.",
                }
        except (ValueError, TypeError):
            return {
                "status": "ERROR",
                "error": f"Invalid places_count: '{places_count}'. Must be an integer.",
            }

        # Validate strings
        if not destination or not isinstance(destination, str) or not destination.strip():
            return {"status": "ERROR", "error": "Destination must be a non-empty string."}
        if not travel_mode or not isinstance(travel_mode, str) or not travel_mode.strip():
            return {"status": "ERROR", "error": "Travel_Mode must be a non-empty string."}
        if not accommodation_type or not isinstance(accommodation_type, str) or not accommodation_type.strip():
            return {"status": "ERROR", "error": "Accommodation_Type must be a non-empty string."}

        # Normalize string cases to title case for matching known training classes
        dest_clean = destination.strip().title()
        mode_clean = travel_mode.strip().title()
        acc_clean = accommodation_type.strip().title()

        # Format input DataFrame
        input_df = format_input_record(
            destination=dest_clean,
            no_of_days=days,
            no_of_travelers=travelers,
            travel_mode=mode_clean,
            accommodation_type=acc_clean,
            places_count=places,
        )

        try:
            raw_prediction = float(self.pipeline.predict(input_df)[0])
            predicted_cost = max(0, int(round(raw_prediction)))

            return {
                "predicted_total_cost": predicted_cost,
                "currency": "INR",
                "model_version": "v2",
                "status": "SUCCESS",
            }
        except Exception as ex:
            return {"status": "ERROR", "error": f"Prediction pipeline error: {str(ex)}"}
