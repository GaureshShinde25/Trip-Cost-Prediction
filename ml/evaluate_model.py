"""
Trip Cost Predictor — Model Evaluation Script
==============================================
Evaluates models/trip_budget_model_v2.pkl against holdout test data and prints
comprehensive metrics: MAE, RMSE, R², MAPE, 5-Fold CV summary, and 10 sample predictions.

Notice:
Model trained on project synthetic dataset.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from ml.train_model import calculate_mape

DATASET_PATH = PROJECT_ROOT / "data" / "trip_budget_dataset_35_destinations.csv"
MODEL_V2_PATH = PROJECT_ROOT / "models" / "trip_budget_model_v2.pkl"


def evaluate():
    if not MODEL_V2_PATH.exists():
        print(f"Model file not found at {MODEL_V2_PATH}. Please run train_model.py first.")
        sys.exit(1)

    print("=" * 75)
    print(" TRIP COST PREDICTOR — V2 MODEL EVALUATION REPORT")
    print(" (Note: Model trained on project synthetic dataset)")
    print("=" * 75)

    # 1. Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset Size:     {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Feature Columns:  {FEATURE_COLUMNS}")
    print(f"Target Column:    {TARGET_COLUMN}")

    # 2. Split (reproducing exact 80/20 test split)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Load model
    pipeline = joblib.load(MODEL_V2_PATH)
    regressor_name = type(pipeline.named_steps["regressor"]).__name__
    print(f"Model Pipeline:   {regressor_name} with ColumnTransformer Preprocessor")

    # 4. Predict on holdout test set
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = calculate_mape(y_test.values, preds)

    print("\n" + "-" * 75)
    print(" TEST SET PERFORMANCE METRICS (20% Holdout = 700 samples)")
    print("-" * 75)
    print(f"  Mean Absolute Error (MAE):           Rs. {mae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE):      Rs. {rmse:,.2f}")
    print(f"  Coefficient of Determination (R²):   {r2:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # 5. 5-Fold Cross-Validation on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")

    print("\n" + "-" * 75)
    print(" 5-FOLD CROSS-VALIDATION RESULTS (Training Set = 2,800 samples)")
    print("-" * 75)
    print(f"  Mean CV MAE:    Rs. {cv_mae.mean():,.2f} (std: ± Rs. {cv_mae.std():,.2f})")
    print(f"  Mean CV R²:     {cv_r2.mean():.4f} (std: ± {cv_r2.std():.4f})")

    # 6. Sample Actual vs Predicted comparisons (10 samples)
    print("\n" + "-" * 75)
    print(" ACTUAL VS PREDICTED SAMPLES (10 Test Samples)")
    print("-" * 75)
    sample_indices = np.random.RandomState(42).choice(len(X_test), size=10, replace=False)

    sample_X = X_test.iloc[sample_indices].reset_index(drop=True)
    sample_y = y_test.iloc[sample_indices].reset_index(drop=True)
    sample_pred = preds[sample_indices]

    print(f"{'#':<3} | {'Destination':<14} | {'Days':<4} | {'Ppl':<3} | {'Mode':<7} | {'Accommodation':<15} | {'Actual (Rs.)':<12} | {'Pred (Rs.)':<12} | {'Diff (%)':<8}")
    print("-" * 75)
    for i in range(10):
        actual = int(sample_y.iloc[i])
        pred = int(round(sample_pred[i]))
        pct_diff = abs(actual - pred) / actual * 100
        dest = str(sample_X.iloc[i]['Destination'])
        days = int(sample_X.iloc[i]['No_of_Days'])
        travelers = int(sample_X.iloc[i]['No_of_Travelers'])
        mode = str(sample_X.iloc[i]['Travel_Mode'])
        acc = str(sample_X.iloc[i]['Accommodation_Type'])

        print(f"{i+1:<3} | {dest:<14} | {days:<4} | {travelers:<3} | {mode:<7} | {acc:<15} | Rs. {actual:<8,} | Rs. {pred:<8,} | {pct_diff:<6.2f}%")
    print("-" * 75 + "\n")


if __name__ == "__main__":
    evaluate()
