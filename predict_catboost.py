"""
Prediction script for the Ames Housing CatBoost model.

This script loads the trained CatBoost model and preprocessor,
applies the same preprocessing steps to the test data,
and generates predictions in the submission format.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Caps numeric features using the 1.5 * IQR rule.
    
    This follows the preprocessing handbook's recommendation to tame outliers
    before fitting models.
    """

    def __init__(self) -> None:
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> "IQRClipper":
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("IQRClipper must be fitted before calling transform.")
        return np.clip(X, self.lower_, self.upper_)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log(1 + x) transformation to reduce skewness in numeric features.
    
    Only transforms features with high skewness (> 0.75) to improve model performance.
    """

    def __init__(self, skew_threshold: float = 0.75) -> None:
        self.skew_threshold = skew_threshold
        self.features_to_transform_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> "LogTransformer":
        # Check skewness for each feature
        # Skip features with very low variance to avoid numerical precision warnings
        self.features_to_transform_ = np.array([
            abs(stats.skew(X[:, i], nan_policy="omit")) > self.skew_threshold
            if np.nanstd(X[:, i]) > 1e-6  # Only check skewness if feature has variance
            else False
            for i in range(X.shape[1])
        ])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.features_to_transform_ is None:
            raise RuntimeError("LogTransformer must be fitted before calling transform.")
        X_transformed = X.copy()
        # Apply log(1 + x) to skewed features (only to positive values)
        for i in range(X.shape[1]):
            if self.features_to_transform_[i]:
                # Only transform non-negative values
                mask = X[:, i] >= 0
                X_transformed[mask, i] = np.log1p(X[mask, i])
        return X_transformed

# Paths
DATA_DIR = Path(__file__).parent
TEST_CSV = DATA_DIR / "test.csv"
MODEL_DIR = DATA_DIR / "models"
MODEL_FILE = MODEL_DIR / "catboost_model.cbm"
PREPROCESSOR_FILE = MODEL_DIR / "catboost_preprocessor.pkl"
METADATA_FILE = MODEL_DIR / "catboost_metadata.pkl"
SUBMISSION_FILE = DATA_DIR / "submission.csv"

# Columns that were dropped during training
LOW_VALUE_COLS: list[str] = [
    "Utilities",
    "Street",
    "Alley",
    "Condition2",
    "RoofMatl",
    "PoolQC",
    "PoolArea",
    "MiscFeature",
    "MiscVal",
    "MoSold",
    "YrSold",
    "LotConfig",
    "LandSlope",
]


def build_feature_sets(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Create numeric and categorical feature lists.
    
    This matches the logic from the training script.
    """
    available_cols = set(features.columns)

    # All object-type columns are categorical
    categorical_cols = [
        col
        for col in features.select_dtypes(include=["object"]).columns.tolist()
        if col in available_cols
    ]

    # Numeric columns (excluding SalePrice if present)
    numeric_cols = [
        col
        for col in features.select_dtypes(include=["number"]).columns.tolist()
        if col not in {"SalePrice"} and col in available_cols
    ]

    # MSSubClass is technically numeric but treated as categorical
    if "MSSubClass" in numeric_cols:
        numeric_cols.remove("MSSubClass")
        if "MSSubClass" in available_cols:
            categorical_cols.append("MSSubClass")

    return numeric_cols, categorical_cols


def main() -> None:
    print("=" * 60)
    print("Loading model and preprocessing artifacts...")
    print("=" * 60)
    
    # Load model
    model = CatBoostRegressor()
    model.load_model(str(MODEL_FILE))
    print(f"  [OK] Model loaded from: {MODEL_FILE}")
    
    # Load preprocessor
    with open(PREPROCESSOR_FILE, "rb") as f:
        preprocessor = pickle.load(f)
    print(f"  [OK] Preprocessor loaded from: {PREPROCESSOR_FILE}")
    
    # Load metadata
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    print(f"  [OK] Metadata loaded from: {METADATA_FILE}")
    
    # Extract metadata
    numeric_cols = metadata["numeric_cols"]
    categorical_cols = metadata["categorical_cols"]
    cat_feature_indices = metadata["cat_feature_indices"]
    
    print(f"\nModel info:")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Total features: {len(numeric_cols) + len(categorical_cols)}")
    
    print("\n" + "=" * 60)
    print("Loading and preprocessing test data...")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    print(f"  [OK] Test data loaded: {test_df.shape[0]} samples, {test_df.shape[1]} columns")
    
    # Save Id column for submission
    test_ids = test_df["Id"].copy()
    
    # Drop the same columns that were dropped during training
    drop_cols = [col for col in LOW_VALUE_COLS if col in test_df.columns]
    drop_cols.append("Id")
    test_df = test_df.drop(columns=drop_cols)
    print(f"  [OK] Dropped {len(drop_cols)} columns")
    
    # Ensure we have the same columns as training (in the same order)
    # Add any missing columns with NaN values
    expected_cols = numeric_cols + categorical_cols
    missing_cols = set(expected_cols) - set(test_df.columns)
    if missing_cols:
        print(f"  ⚠ Warning: Missing columns in test data: {missing_cols}")
        for col in missing_cols:
            test_df[col] = np.nan
    
    # Reorder columns to match training order
    test_df = test_df[expected_cols]
    
    # Preprocess numeric features
    print("\n  Preprocessing numeric features...")
    test_numeric = preprocessor.transform(test_df[numeric_cols])
    print(f"    [OK] Numeric features preprocessed: {test_numeric.shape}")
    
    # Handle categorical features
    print("  Preprocessing categorical features...")
    test_categorical = test_df[categorical_cols].copy()
    # Fill missing values with empty string (CatBoost handles this)
    test_categorical = test_categorical.fillna("")
    print(f"    [OK] Categorical features preprocessed: {test_categorical.shape}")
    
    # Combine numeric and categorical features
    # Numeric features come first, then categoricals (matching training)
    test_combined = np.hstack([test_numeric, test_categorical.values])
    print(f"    [OK] Combined features shape: {test_combined.shape}")
    
    print("\n" + "=" * 60)
    print("Making predictions...")
    print("=" * 60)
    
    # Make predictions
    predictions = model.predict(test_combined)
    print(f"  [OK] Generated {len(predictions)} predictions")
    print(f"    Prediction range: ${predictions.min():,.2f} - ${predictions.max():,.2f}")
    print(f"    Mean prediction: ${predictions.mean():,.2f}")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions
    })
    
    # Save submission
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n  [OK] Submission saved to: {SUBMISSION_FILE}")
    
    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)
    print(f"\nSubmission file: {SUBMISSION_FILE}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

