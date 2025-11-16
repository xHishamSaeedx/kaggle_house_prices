"""
Training script for the Ames Housing CatBoost modeling exercise.

The implementation leverages CatBoost's native categorical feature handling:
- CatBoost automatically handles categorical encoding using Ordered Target Statistics
- This prevents data leakage and overfitting without manual preprocessing
- Only numeric features need preprocessing (imputation, outlier capping)
- Categorical features are passed directly to CatBoost

This follows the provided preprocessing and modeling rulebooks:
- `preprocessing_rulebook.md` guides missing-value handling and outlier capping for numerics
- `column_analysis.md` enumerates the columns to keep vs. safely drop.
- CatBoost is ideal for this dataset due to its many categorical features (43 out of 81 columns).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline


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


DATA_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train.csv"

# Columns that the column analysis deemed low-utility or noisy for SalePrice.
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


def build_feature_sets(features: pd.DataFrame) -> tuple[List[str], List[str]]:
    """
    Create numeric and categorical feature lists.
    
    CatBoost handles all categorical features natively, so we don't need to
    distinguish between ordinal and nominal categoricals.
    """
    available_cols = set(features.columns)

    # All object-type columns are categorical
    categorical_cols = [
        col
        for col in features.select_dtypes(include=["object"]).columns.tolist()
        if col in available_cols
    ]

    # Numeric columns (excluding SalePrice)
    numeric_cols = [
        col
        for col in features.select_dtypes(include=["number"]).columns.tolist()
        if col not in {"SalePrice"} and col in available_cols
    ]

    # MSSubClass is technically numeric but treated as categorical per the column analysis
    if "MSSubClass" in numeric_cols:
        numeric_cols.remove("MSSubClass")
        if "MSSubClass" in available_cols:
            categorical_cols.append("MSSubClass")

    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """
    Construct a simplified ColumnTransformer for numeric features only.
    
    CatBoost handles categorical features natively, so we only need to
    preprocess numeric features (imputation and outlier capping).
    """
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("outlier_capper", IQRClipper()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)

    drop_cols = [col for col in LOW_VALUE_COLS if col in df.columns]
    drop_cols.append("Id")
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_cols, categorical_cols = build_feature_sets(X)

    # Preprocess numeric features only (imputation and outlier capping)
    preprocessor = make_preprocessor(numeric_cols)
    
    # Transform numeric features
    X_train_numeric = preprocessor.fit_transform(X_train[numeric_cols])
    X_valid_numeric = preprocessor.transform(X_valid[numeric_cols])
    
    # Keep categorical features as-is (CatBoost handles them natively)
    X_train_categorical = X_train[categorical_cols].copy()
    X_valid_categorical = X_valid[categorical_cols].copy()
    
    # Fill missing values in categoricals with empty string (CatBoost handles this)
    X_train_categorical = X_train_categorical.fillna("")
    X_valid_categorical = X_valid_categorical.fillna("")
    
    # Combine numeric and categorical features
    # Numeric features come first, then categoricals
    X_train_combined = np.hstack([X_train_numeric, X_train_categorical.values])
    X_valid_combined = np.hstack([X_valid_numeric, X_valid_categorical.values])
    
    # Get categorical feature indices (they come after numeric features)
    n_numeric = len(numeric_cols)
    cat_feature_indices = list(range(n_numeric, n_numeric + len(categorical_cols)))

    # CatBoost hyperparameter grid (simpler than XGBoost due to strong defaults)
    param_grid = {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = float("inf")
    best_params = None

    # Manual grid search to handle categorical features properly
    print("Starting grid search...")
    total_combinations = (
        len(param_grid["iterations"])
        * len(param_grid["depth"])
        * len(param_grid["learning_rate"])
        * len(param_grid["l2_leaf_reg"])
    )
    current = 0

    for iterations in param_grid["iterations"]:
        for depth in param_grid["depth"]:
            for learning_rate in param_grid["learning_rate"]:
                for l2_leaf_reg in param_grid["l2_leaf_reg"]:
                    current += 1
                    cv_scores = []
                    for train_idx, val_idx in cv.split(X_train_combined):
                        X_cv_train, X_cv_val = (
                            X_train_combined[train_idx],
                            X_train_combined[val_idx],
                        )
                        y_cv_train, y_cv_val = (
                            y_train.iloc[train_idx].values,
                            y_train.iloc[val_idx].values,
                        )

                        model = CatBoostRegressor(
                            iterations=iterations,
                            depth=depth,
                            learning_rate=learning_rate,
                            l2_leaf_reg=l2_leaf_reg,
                            cat_features=cat_feature_indices,
                            random_seed=42,
                            verbose=False,
                        )
                        model.fit(X_cv_train, y_cv_train)
                        preds = model.predict(X_cv_val)
                        rmse = mean_squared_error(y_cv_val, preds) ** 0.5
                        cv_scores.append(rmse)

                    avg_rmse = np.mean(cv_scores)
                    if avg_rmse < best_score:
                        best_score = avg_rmse
                        best_params = {
                            "iterations": iterations,
                            "depth": depth,
                            "learning_rate": learning_rate,
                            "l2_leaf_reg": l2_leaf_reg,
                        }
                        print(
                            f"[{current}/{total_combinations}] New best CV RMSE: {avg_rmse:,.2f} with params: {best_params}"
                        )

    print(f"\nBest parameters: {best_params}")
    print(f"Cross-validated RMSE: {best_score:,.2f}")

    # Train final model with best parameters
    best_model = CatBoostRegressor(
        **best_params,
        cat_features=cat_feature_indices,
        random_seed=42,
        verbose=False,
    )
    best_model.fit(X_train_combined, y_train.values)

    val_preds = best_model.predict(X_valid_combined)
    val_rmse = mean_squared_error(y_valid, val_preds) ** 0.5
    val_mae = mean_absolute_error(y_valid, val_preds)
    val_r2 = r2_score(y_valid, val_preds)
    print(f"\nValidation RMSE: {val_rmse:,.2f}")
    print(f"Validation MAE: {val_mae:,.2f}")
    print(f"Validation R^2 (accuracy proxy): {val_r2:.4f}")

    train_preds = best_model.predict(X_train_combined)
    train_rmse = mean_squared_error(y_train, train_preds) ** 0.5
    print(f"Training RMSE (sanity check): {train_rmse:,.2f}")


if __name__ == "__main__":
    main()
