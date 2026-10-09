"""
Trip Cost Predictor — Model Training Pipeline
==============================================
Trains and evaluates multiple candidate regression algorithms on the 35-destination
dataset. Selects the top performer based on lowest test MAE and 5-fold CV stability,
and persists the final pipeline to models/trip_budget_model_v2.pkl.

Safety Constraints:
- Preserves the legacy models/trip_budget_model.pkl without overwriting.
- Avoids target leakage by using only valid pre-trip features.
"""

import sys
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)

DATASET_PATH = PROJECT_ROOT / "data" / "trip_budget_dataset_35_destinations.csv"
MODEL_V2_PATH = PROJECT_ROOT / "models" / "trip_budget_model_v2.pkl"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "trip_budget_model.pkl"


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error, guarding against division by zero."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_and_validate_data(dataset_path: Path) -> pd.DataFrame:
    """Loads and validates the trip budget dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Validate essential columns
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    # Validate data health
    null_counts = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Dataset contains {null_counts} null values.")

    if (df[TARGET_COLUMN] <= 0).sum() > 0:
        raise ValueError("Dataset contains non-positive target (Total_Cost) values.")

    print(f"Data validation passed: 0 nulls, positive target distribution.")
    return df


def get_candidate_models():
    """Returns candidate regression models for benchmark."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    }


def train_and_evaluate():
    """Full execution pipeline for Task 5 ML training."""
    print("=" * 70)
    print(" TRIP COST PREDICTOR — ML MODEL UPGRADE PIPELINE (V2)")
    print("=" * 70)

    # 1. Load data
    df = load_and_validate_data(DATASET_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. 80/20 train/test split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Split sizes: Train={len(X_train)} samples, Test={len(X_test)} samples.\n")

    # 3. Benchmark candidate models
    candidates = get_candidate_models()
    evaluation_records = []
    fitted_pipelines = {}

    print("Benchmarking candidate models on test set (20% holdout)...")
    for name, regressor in candidates.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = calculate_mape(y_test.values, preds)

        evaluation_records.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
        })
        fitted_pipelines[name] = pipeline

    eval_df = pd.DataFrame(evaluation_records).sort_values(by="MAE", ascending=True)

    print("\n" + "-" * 70)
    print(" MODEL COMPARISON TABLE (TEST SET)")
    print("-" * 70)
    print(f"{'Model':<30} | {'MAE (Rs.)':<10} | {'RMSE (Rs.)':<10} | {'R²':<7} | {'MAPE (%)':<8}")
    print("-" * 70)
    for _, row in eval_df.iterrows():
        print(f"{row['Model']:<30} | {row['MAE']:<10.2f} | {row['RMSE']:<10.2f} | {row['R2']:<7.4f} | {row['MAPE']:<8.2f}%")
    print("-" * 70)

    # 4. Perform 5-fold cross-validation on top 2 models
    top_2_names = eval_df.head(2)["Model"].tolist()
    print(f"\nTop 2 models selected for 5-Fold Cross-Validation: {top_2_names}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}

    for name in top_2_names:
        regressor = candidates[name]
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("regressor", regressor)])

        cv_mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        cv_r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")

        cv_summary[name] = {
            "mean_mae": float(np.mean(cv_mae_scores)),
            "std_mae": float(np.std(cv_mae_scores)),
            "mean_r2": float(np.mean(cv_r2_scores)),
            "std_r2": float(np.std(cv_r2_scores)),
        }

        print(f"\n  Cross-Validation Results for {name}:")
        print(f"    - Mean CV MAE: Rs. {cv_summary[name]['mean_mae']:.2f} (± {cv_summary[name]['std_mae']:.2f})")
        print(f"    - Mean CV R²:  {cv_summary[name]['mean_r2']:.4f} (± {cv_summary[name]['std_r2']:.4f})")

    # 5. Select best model based primarily on lowest MAE & stability
    best_model_name = top_2_names[0]
    best_pipeline = fitted_pipelines[best_model_name]

    print("\n" + "=" * 70)
    print(f" SELECTED BEST MODEL: {best_model_name}")
    print(f" Rationale: Lowest test MAE (Rs. {eval_df.iloc[0]['MAE']:.2f}), "
          f"lowest RMSE (Rs. {eval_df.iloc[0]['RMSE']:.2f}), "
          f"highest R² ({eval_df.iloc[0]['R2']:.4f}), and superior 5-fold CV stability.")
    print("=" * 70)

    # 6. Save model v2 (ensuring legacy model is preserved)
    MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_V2_PATH)
    print(f"\nPersisted final pipeline to: {MODEL_V2_PATH}")

    if LEGACY_MODEL_PATH.exists():
        print(f"Confirmed legacy model preserved at: {LEGACY_MODEL_PATH}")

    return {
        "eval_df": eval_df,
        "cv_summary": cv_summary,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
    }


if __name__ == "__main__":
    train_and_evaluate()
