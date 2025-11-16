"""
Training script for the Ames Housing linear modeling exercise.

The implementation follows the provided preprocessing and modeling rulebooks:
- `preprocessing_rulebook.md` guides missing-value handling, outlier capping,
  feature scaling, and categorical encoding.
- `column_analysis.md` enumerates the columns to keep vs. safely drop.
- `regression_model_rulebook.md` points us to Ridge regression because the
  dataset has many moderately correlated predictors and we want a stable model
  without aggressive feature selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Caps numeric features using the 1.5 * IQR rule.

    This follows the preprocessing handbook's recommendation to tame outliers
    before fitting OLS-family models.
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
LOW_VALUE_COLS: Sequence[str] = [
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

# Ordinal categories ordered from worst to best.
QUALITY_MAP = ["Po", "Fa", "TA", "Gd", "Ex"]
BSMT_EXPOSURE = ["No", "Mn", "Av", "Gd"]
BSMT_FINISHED = ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
GARAGE_FINISH = ["Unf", "RFn", "Fin"]
FENCE_QUALITY = ["MnWw", "GdWo", "MnPrv", "GdPrv"]
CENTRAL_AIR = ["N", "Y"]
FIREPLACE_QUAL = ["Po", "Fa", "TA", "Gd", "Ex"]

ORDINAL_CONFIG = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "BsmtExposure": BSMT_EXPOSURE,
    "BsmtFinType1": BSMT_FINISHED,
    "BsmtFinType2": BSMT_FINISHED,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": FIREPLACE_QUAL,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "GarageFinish": GARAGE_FINISH,
    "Fence": FENCE_QUALITY,
    "CentralAir": CENTRAL_AIR,
}


def build_feature_sets(features: pd.DataFrame) -> tuple[List[str], List[str], List[str]]:
    """Create numeric, ordinal, and nominal feature lists based on available columns."""
    available_cols = set(features.columns)

    ordinal_cols = [col for col in ORDINAL_CONFIG if col in available_cols]

    # Structural and other nominal categoricals that benefit from one-hot encoding.
    structural_cats = [
        "MSSubClass",
        "MSZoning",
        "LotShape",
        "LandContour",
        "Neighborhood",
        "Condition1",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Foundation",
        "Heating",
        "Electrical",
        "Functional",
        "GarageType",
        "GarageCond",
        "GarageQual",
        "PavedDrive",
        "SaleType",
        "SaleCondition",
    ]
    categorical_cols = [col for col in structural_cats if col in available_cols and col not in ordinal_cols]

    numeric_cols = [
        col
        for col in features.select_dtypes(include=["number"]).columns.tolist()
        if col not in {"SalePrice"}
    ]
    # MSSubClass is technically numeric but treated as categorical per the column analysis.
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols and col not in ordinal_cols]

    return numeric_cols, ordinal_cols, categorical_cols


def make_preprocessor(
    numeric_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """Construct the ColumnTransformer that executes the preprocessing handbook."""

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("outlier_capper", IQRClipper()),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if ordinal_cols:
        ordinal_categories = [ORDINAL_CONFIG[col] for col in ordinal_cols]
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=ordinal_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


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

    numeric_cols, ordinal_cols, categorical_cols = build_feature_sets(X)
    preprocessor = make_preprocessor(numeric_cols, ordinal_cols, categorical_cols)

    ridge = Ridge(random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", ridge),
        ]
    )

    # Ridge is recommended by the regression rulebook for many correlated features.
    param_grid = {"model__alpha": np.logspace(-2, 3, 20)}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_rmse = -search.best_score_
    print(f"Best alpha: {search.best_params_['model__alpha']:.4f}")
    print(f"Cross-validated RMSE: {best_rmse:,.2f}")

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    val_preds = best_model.predict(X_valid)
    val_rmse = mean_squared_error(y_valid, val_preds) ** 0.5
    val_mae = mean_absolute_error(y_valid, val_preds)
    val_r2 = r2_score(y_valid, val_preds)
    print(f"Validation RMSE: {val_rmse:,.2f}")
    print(f"Validation MAE: {val_mae:,.2f}")
    print(f"Validation R^2 (accuracy proxy): {val_r2:.4f}")

    train_preds = best_model.predict(X_train)
    train_rmse = mean_squared_error(y_train, train_preds) ** 0.5
    print(f"Training RMSE (sanity check): {train_rmse:,.2f}")


if __name__ == "__main__":
    main()

