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

import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

# Try to import GPU device count function (may not be available in all versions)
try:
    from catboost import get_gpu_device_count
    HAS_GPU_COUNT_FUNC = True
except ImportError:
    HAS_GPU_COUNT_FUNC = False


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


DATA_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train.csv"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "catboost_model.cbm"
PREPROCESSOR_FILE = MODEL_DIR / "catboost_preprocessor.pkl"
METADATA_FILE = MODEL_DIR / "catboost_metadata.pkl"

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


def make_preprocessor(numeric_cols: list[str], use_log_transform: bool = True) -> ColumnTransformer:
    """
    Construct a simplified ColumnTransformer for numeric features only.
    
    CatBoost handles categorical features natively, so we only need to
    preprocess numeric features (imputation, outlier capping, and optional log transform).
    
    Args:
        numeric_cols: List of numeric column names
        use_log_transform: Whether to apply log transformation to skewed features
    """
    transformers = []

    if numeric_cols:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_capper", IQRClipper()),
        ]
        if use_log_transform:
            steps.append(("log_transform", LogTransformer(skew_threshold=0.75)))
        
        numeric_pipeline = Pipeline(steps=steps)
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def plot_training_history(model: CatBoostRegressor, save_path: Path) -> None:
    """
    Plot training and validation loss curves from CatBoost training history.
    """
    try:
        # Get training history
        history = model.get_evals_result()
        
        if not history:
            print("  ⚠ No training history available for plotting")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training and Validation RMSE
        ax1 = axes[0]
        if "learn" in history:
            train_rmse = history["learn"]["RMSE"]
            ax1.plot(train_rmse, label="Train RMSE", linewidth=2)
        
        if "validation" in history:
            val_rmse = history["validation"]["RMSE"]
            ax1.plot(val_rmse, label="Validation RMSE", linewidth=2)
        
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("RMSE", fontsize=12)
        ax1.set_title("Training and Validation RMSE", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RMSE Improvement
        ax2 = axes[1]
        if "learn" in history and "validation" in history:
            train_rmse = history["learn"]["RMSE"]
            val_rmse = history["validation"]["RMSE"]
            iterations = range(len(train_rmse))
            
            ax2.plot(iterations, train_rmse, label="Train RMSE", linewidth=2, alpha=0.7)
            ax2.plot(iterations, val_rmse, label="Validation RMSE", linewidth=2, alpha=0.7)
            ax2.fill_between(iterations, train_rmse, val_rmse, alpha=0.2, label="Gap")
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("RMSE", fontsize=12)
            ax2.set_title("RMSE Convergence", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Training history plot saved to: {save_path}")
    except Exception as e:
        print(f"  ⚠ Could not create training history plot: {e}")


def plot_feature_importance(importance_df: pd.DataFrame, save_path: Path, top_n: int = 20) -> None:
    """
    Plot feature importance as a horizontal bar chart.
    """
    try:
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        y_pos = np.arange(len(top_features))
        
        bars = ax.barh(y_pos, top_features["importance"], align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features["feature"], fontsize=10)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row["importance"] * 0.01, i, f"{row['importance']:.0f}", 
                   va="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Feature importance plot saved to: {save_path}")
    except Exception as e:
        print(f"  ⚠ Could not create feature importance plot: {e}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str = "Predictions") -> None:
    """
    Plot predicted vs actual values with residual analysis.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Predicted vs Actual
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
        
        ax1.set_xlabel("Actual SalePrice", fontsize=12)
        ax1.set_ylabel("Predicted SalePrice", fontsize=12)
        ax1.set_title(f"{title}: Predicted vs Actual", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Calculate R² for display
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        ax1.text(0.05, 0.95, f"R² = {r2:.4f}\nRMSE = ${rmse:,.0f}", 
                transform=ax1.transAxes, fontsize=11,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        # Plot 2: Residuals
        ax2 = axes[1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("Predicted SalePrice", fontsize=12)
        ax2.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
        ax2.set_title(f"{title}: Residual Plot", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Prediction plot saved to: {save_path}")
    except Exception as e:
        print(f"  ⚠ Could not create prediction plot: {e}")


def check_device_availability() -> tuple[str, bool]:
    """
    Check if GPU is available for CatBoost.
    
    Returns:
        Tuple of (device_name, is_available)
    """
    # Method 1: Try using get_gpu_device_count if available
    if HAS_GPU_COUNT_FUNC:
        try:
            gpu_count = get_gpu_device_count()
            if gpu_count > 0:
                return "GPU", True
        except Exception:
            pass
    
    # Method 2: Try creating a small test model with GPU to detect availability
    try:
        # Create a tiny test dataset
        X_test = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y_test = np.array([1, 2], dtype=np.float32)
        
        # Try to create and fit a model with GPU
        test_model = CatBoostRegressor(
            iterations=1,
            task_type="GPU",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        test_model.fit(X_test, y_test)
        return "GPU", True
    except Exception:
        # GPU not available, will use CPU
        return "CPU", False


def main() -> None:
    # Check and log device availability
    device_name, gpu_available = check_device_availability()
    if gpu_available:
        if HAS_GPU_COUNT_FUNC:
            try:
                gpu_count = get_gpu_device_count()
                print(f"✓ GPU detected: {gpu_count} device(s) available")
            except Exception:
                print("✓ GPU detected: available")
        else:
            print("✓ GPU detected: available")
        print(f"  Using device: GPU\n")
    else:
        print("⚠ GPU not available, will use CPU")
        print("  (CatBoost will automatically fall back to CPU if GPU is unavailable)\n")
    
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
    
    print(f"Dataset shape: {X.shape}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Total features: {len(numeric_cols) + len(categorical_cols)}\n")

    # Preprocess numeric features (imputation, outlier capping, and log transform for skewed features)
    preprocessor = make_preprocessor(numeric_cols, use_log_transform=True)
    
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

    # CatBoost hyperparameter grid - optimized for better generalization
    # Adjusted based on overfitting analysis:
    # - Lower depth to reduce complexity
    # - Higher l2_leaf_reg for stronger regularization
    # - Lower learning rates for smoother convergence
    # - More iterations with early stopping
    param_grid = {
        "iterations": [300, 500, 700],  # More iterations, rely on early stopping
        "depth": [3, 4, 5],  # Reduced from [4,6,8] to prevent overfitting
        "learning_rate": [0.01, 0.03, 0.05],  # Lower learning rates for stability
        "l2_leaf_reg": [5, 10, 15],  # Increased regularization
        "min_data_in_leaf": [5, 10, 20],  # Additional regularization
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = float("inf")
    best_params = None

    # Manual grid search to handle categorical features properly
    print(f"Starting grid search (using {device_name})...")
    total_combinations = (
        len(param_grid["iterations"])
        * len(param_grid["depth"])
        * len(param_grid["learning_rate"])
        * len(param_grid["l2_leaf_reg"])
        * len(param_grid["min_data_in_leaf"])
    )
    current = 0

    for iterations in param_grid["iterations"]:
        for depth in param_grid["depth"]:
            for learning_rate in param_grid["learning_rate"]:
                for l2_leaf_reg in param_grid["l2_leaf_reg"]:
                    for min_data_in_leaf in param_grid["min_data_in_leaf"]:
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
                                min_data_in_leaf=min_data_in_leaf,
                                cat_features=cat_feature_indices,
                                task_type="GPU",
                                random_seed=42,
                                verbose=False,
                                early_stopping_rounds=100,  # Increased from 50 for better stopping
                                eval_metric="RMSE",
                                bootstrap_type="Bernoulli",  # Change bootstrap type to support subsample
                                subsample=0.8,  # Add bagging to reduce overfitting
                                # colsample_bylevel not supported on GPU for regression
                            )
                            # Use eval_set for early stopping and monitoring
                            model.fit(
                                X_cv_train,
                                y_cv_train,
                                eval_set=(X_cv_val, y_cv_val),
                                verbose=False,
                            )
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
                                "min_data_in_leaf": min_data_in_leaf,
                            }
                            print(
                                f"[{current}/{total_combinations}] New best CV RMSE: {avg_rmse:,.2f} with params: {best_params}"
                            )

    print(f"\nBest parameters: {best_params}")
    print(f"Cross-validated RMSE: {best_score:,.2f}")

    # Train final model with best parameters
    print(f"\nTraining final model (using {device_name})...")
    best_model = CatBoostRegressor(
        **best_params,
        cat_features=cat_feature_indices,
        task_type="GPU",
        random_seed=42,
        verbose=100,  # Show progress every 100 iterations
        early_stopping_rounds=100,  # Increased for better generalization
        eval_metric="RMSE",
        bootstrap_type="Bernoulli",  # Change bootstrap type to support subsample
        subsample=0.8,  # Add bagging to reduce overfitting
        # colsample_bylevel not supported on GPU for regression
    )
    # Use validation set for early stopping and monitoring
    best_model.fit(
        X_train_combined,
        y_train.values,
        eval_set=(X_valid_combined, y_valid.values),
        verbose=100,
    )
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # Plot training history
    plot_training_history(best_model, PLOTS_DIR / "training_history.png")
    
    # Display feature importance
    print("\n" + "="*60)
    print("Top 20 Most Important Features:")
    print("="*60)
    feature_importance = best_model.get_feature_importance()
    feature_names = numeric_cols + categorical_cols
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)
    print(importance_df.head(20).to_string(index=False))
    print("="*60)

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

    # Save model and preprocessing artifacts for later use
    print(f"\nSaving model and preprocessing artifacts...")
    best_model.save_model(str(MODEL_FILE))
    print(f"  ✓ Model saved to: {MODEL_FILE}")
    
    with open(PREPROCESSOR_FILE, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"  ✓ Preprocessor saved to: {PREPROCESSOR_FILE}")
    
    # Save feature importance
    importance_df.to_csv(MODEL_DIR / "catboost_feature_importance.csv", index=False)
    print(f"  ✓ Feature importance saved to: {MODEL_DIR / 'catboost_feature_importance.csv'}")
    
    # Plot feature importance
    plot_feature_importance(importance_df, PLOTS_DIR / "feature_importance.png", top_n=20)
    
    # Plot predictions
    plot_predictions(y_valid.values, val_preds, PLOTS_DIR / "validation_predictions.png", title="Validation")
    plot_predictions(y_train.values, train_preds, PLOTS_DIR / "train_predictions.png", title="Training")
    
    # Save metadata needed for prediction
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_feature_indices": cat_feature_indices,
        "low_value_cols": LOW_VALUE_COLS,
        "best_params": best_params,
        "validation_rmse": val_rmse,
        "validation_mae": val_mae,
        "validation_r2": val_r2,
        "train_rmse": train_rmse,
    }
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Metadata saved to: {METADATA_FILE}")
    
    # Print overfitting check
    overfit_ratio = train_rmse / val_rmse if val_rmse > 0 else float("inf")
    print(f"\n{'='*60}")
    print("Overfitting Check:")
    print(f"  Train RMSE / Val RMSE ratio: {overfit_ratio:.4f}")
    if overfit_ratio < 0.9:
        print("  ⚠ Warning: Possible overfitting (train RMSE much lower than validation)")
    elif overfit_ratio > 1.1:
        print("  ⚠ Warning: Possible underfitting (train RMSE higher than validation)")
    else:
        print("  ✓ Good generalization (train and validation RMSE are similar)")
    print(f"{'='*60}")
    print(f"\nModel artifacts saved successfully! You can now use them for predictions.")
    print(f"\n{'='*60}")
    print("Visualization Summary:")
    print(f"{'='*60}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"  • training_history.png - Training/validation loss curves")
    print(f"  • feature_importance.png - Top 20 most important features")
    print(f"  • validation_predictions.png - Validation set predictions vs actual")
    print(f"  • train_predictions.png - Training set predictions vs actual")
    print(f"\nOpen these files to view the visualizations!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

