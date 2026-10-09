"""
Trip Cost Predictor — ML Preprocessing Pipeline
================================================
Defines feature schema and scikit-learn ColumnTransformer for trip cost prediction.
Supports robust categorical encoding with unseen category tolerance and numerical scaling.
"""

from typing import List, Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature schema definitions
CATEGORICAL_FEATURES: List[str] = ["Destination", "Travel_Mode", "Accommodation_Type"]
NUMERICAL_FEATURES: List[str] = ["No_of_Days", "No_of_Travelers", "Places_Count"]
FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN: str = "Total_Cost"


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs scikit-learn ColumnTransformer.
    - Categorical: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    - Numerical: StandardScaler()
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def format_input_record(
    destination: str,
    no_of_days: int,
    no_of_travelers: int,
    travel_mode: str,
    accommodation_type: str,
    places_count: int,
) -> pd.DataFrame:
    """
    Formats individual prediction parameters into a single-row DataFrame
    matching the trained model's feature schema.
    """
    record = {
        "Destination": str(destination).strip(),
        "No_of_Days": int(no_of_days),
        "No_of_Travelers": int(no_of_travelers),
        "Travel_Mode": str(travel_mode).strip(),
        "Accommodation_Type": str(accommodation_type).strip(),
        "Places_Count": int(places_count),
    }
    return pd.DataFrame([record], columns=FEATURE_COLUMNS)
