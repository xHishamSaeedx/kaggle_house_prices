# Kaggle House Prices - CatBoost Model

This project predicts house prices using the Ames Housing dataset from Kaggle. The main training script (`train_catboost_model.py`) uses CatBoost, a powerful gradient boosting library that excels at handling mixed data types (both numbers and categories).

## Table of Contents

- [Overview](#overview)
- [What is CatBoost and Why Use It?](#what-is-catboost-and-why-use-it)
- [Data Preprocessing](#data-preprocessing)
- [Custom Transformers Explained](#custom-transformers-explained)
- [Model Training Process](#model-training-process)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Visualizations](#visualizations)
- [Output Files](#output-files)

---

## Overview

The script trains a machine learning model to predict house sale prices. Here's the high-level flow:

1. **Load Data**: Reads the training CSV file with house features and prices
2. **Clean Data**: Removes low-value columns that don't help predictions
3. **Split Data**: Divides data into training (80%) and validation (20%) sets
4. **Preprocess**: Cleans and transforms numeric features
5. **Find Best Settings**: Tests many combinations of model settings (hyperparameters)
6. **Train Final Model**: Trains the best model on all training data
7. **Evaluate**: Checks how well the model performs
8. **Save Everything**: Saves the model, preprocessor, and visualizations

---

## What is CatBoost and Why Use It?

**CatBoost** (Categorical Boosting) is a machine learning algorithm that:

- **Handles categories automatically**: Unlike other algorithms, you don't need to manually convert categories (like "Brick", "Wood", "Vinyl") into numbers. CatBoost does this intelligently.
- **Prevents overfitting**: Uses special techniques to avoid memorizing the training data
- **Works with mixed data**: Can use both numeric features (like square footage) and categorical features (like neighborhood) together seamlessly

**Why this dataset is perfect for CatBoost:**

- The dataset has **43 categorical features** out of 81 total columns
- CatBoost's native categorical handling means less preprocessing work
- Better performance on datasets with many categories

---

## Data Preprocessing

Before training, the data needs to be cleaned and prepared. The script does this in several steps:

### Step 1: Remove Low-Value Columns

Some columns don't help predict prices (like "Street" type, "Utilities", etc.). These are dropped:

```python
LOW_VALUE_COLS = ["Utilities", "Street", "Alley", "Condition2", ...]
```

### Step 2: Separate Features by Type

The script splits features into two groups:

- **Numeric features**: Numbers like square footage, number of bedrooms, year built
- **Categorical features**: Text/categories like neighborhood, house style, roof material

### Step 3: Preprocess Numeric Features

Numeric features go through a pipeline of transformations:

1. **Imputation** (Fill Missing Values)

   - Uses median value to fill missing numbers
   - Example: If a house is missing "LotArea", fill it with the median lot area

2. **Outlier Capping** (IQRClipper)

   - Removes extreme values that could confuse the model
   - Uses the "1.5 × IQR rule": values beyond 1.5× the interquartile range are capped
   - Example: If most houses are 1000-2000 sq ft, a 10,000 sq ft house gets capped to a reasonable maximum

3. **Log Transformation** (LogTransformer)
   - For highly skewed features (like prices, which have a long tail)
   - Applies log(1 + x) to make the distribution more normal
   - Only transforms features with skewness > 0.75
   - Example: House prices might be $50k, $100k, $200k, $500k - log transform makes them more evenly spread

### Step 4: Handle Categorical Features

- **CatBoost handles these automatically!** No manual encoding needed
- Missing categorical values are filled with empty strings `""`
- CatBoost uses "Ordered Target Statistics" to encode categories intelligently

### Step 5: Combine Features

- Numeric features (preprocessed) and categorical features (as-is) are combined into one array
- The model needs to know which columns are categorical, so their indices are tracked

---

## Custom Transformers Explained

The script includes two custom transformers:

### 1. IQRClipper (`IQRClipper` class)

**What it does**: Caps extreme values (outliers) in numeric features.

**How it works**:

- Calculates Q1 (25th percentile) and Q3 (75th percentile) for each feature
- Computes IQR (Interquartile Range) = Q3 - Q1
- Sets lower bound = Q1 - 1.5 × IQR
- Sets upper bound = Q3 + 1.5 × IQR
- Any value below the lower bound gets raised to the lower bound
- Any value above the upper bound gets lowered to the upper bound

**Why**: Extreme values can make the model focus on outliers instead of learning general patterns.

**Example**:

```
LotArea values: [5000, 6000, 7000, 8000, 9000, 50000]
Q1 = 6000, Q3 = 8000, IQR = 2000
Lower = 6000 - 1.5×2000 = 3000
Upper = 8000 + 1.5×2000 = 11000
Result: [5000, 6000, 7000, 8000, 9000, 11000]  (50000 capped to 11000)
```

### 2. LogTransformer (`LogTransformer` class)

**What it does**: Applies logarithmic transformation to highly skewed features.

**How it works**:

- Calculates skewness for each numeric feature
- If skewness > 0.75 (threshold), the feature is marked for transformation
- Applies log(1 + x) to non-negative values only
- Uses `np.log1p()` which is numerically stable (handles x=0 correctly)

**Why**: Many real-world features (prices, areas, counts) have skewed distributions. Log transformation makes them more normally distributed, which helps linear models and tree-based models.

**Example**:

```
Original prices: [50000, 100000, 200000, 500000]
After log(1+x): [10.82, 11.51, 12.21, 13.12]
Distribution is now more spread out and normal
```

---

## Model Training Process

### Phase 1: Hyperparameter Tuning (Grid Search)

**What are hyperparameters?**
These are settings that control how the model learns, but aren't learned from data themselves. Think of them as "knobs" you can tune.

**The grid search tests these combinations**:

```python
param_grid = {
    "iterations": [300, 500, 700],      # How many trees to build
    "depth": [3, 4, 5],                  # How deep each tree can be
    "learning_rate": [0.01, 0.03, 0.05], # How fast the model learns
    "l2_leaf_reg": [5, 10, 15],          # Regularization strength
    "min_data_in_leaf": [5, 10, 20],     # Minimum samples per leaf
}
```

**Total combinations**: 3 × 3 × 3 × 3 × 3 = **243 combinations**

**How it works**:

1. For each combination, the script uses **5-fold cross-validation**:
   - Splits training data into 5 parts
   - Trains on 4 parts, tests on 1 part
   - Repeats 5 times (each part used as test once)
   - Averages the 5 scores
2. Tracks the combination with the lowest RMSE (Root Mean Squared Error)
3. This process can take a while (testing 243 combinations × 5 folds = 1,215 model trainings!)

**Why these specific values?**

- **Lower depth (3-5)**: Prevents overfitting (model memorizing training data)
- **Lower learning rates (0.01-0.05)**: More stable, smoother learning
- **Higher regularization (l2_leaf_reg 5-15)**: Reduces overfitting
- **More iterations (300-700)**: With early stopping, the model stops when it stops improving

### Phase 2: Training the Final Model

Once the best hyperparameters are found:

1. Creates a CatBoost model with those best settings
2. Trains on the full training set (80% of data)
3. Uses the validation set (20%) for **early stopping**:
   - Monitors validation error
   - If error doesn't improve for 100 iterations, stops training
   - Prevents overfitting by stopping when the model stops learning

**Additional settings**:

- `subsample=0.8`: Uses 80% of data randomly for each tree (bagging)
- `bootstrap_type="Bernoulli"`: Enables subsampling
- `cat_features=cat_feature_indices`: Tells CatBoost which columns are categorical

---

## Hyperparameter Tuning

### What Each Hyperparameter Does

1. **`iterations`** (300, 500, 700)

   - Number of decision trees to build
   - More trees = more complex model, but risk of overfitting
   - Early stopping will stop early if needed

2. **`depth`** (3, 4, 5)

   - Maximum depth of each decision tree
   - Deeper trees = more complex patterns, but more overfitting risk
   - Reduced from [4,6,8] to prevent overfitting

3. **`learning_rate`** (0.01, 0.03, 0.05)

   - How much each tree contributes to the final prediction
   - Lower = slower learning, more stable
   - Lower rates help prevent overfitting

4. **`l2_leaf_reg`** (5, 10, 15)

   - Regularization parameter (penalty for complexity)
   - Higher values = simpler models, less overfitting
   - Increased from [1,3,5] for stronger regularization

5. **`min_data_in_leaf`** (5, 10, 20)
   - Minimum number of samples required in a leaf node
   - Higher values = simpler trees, less overfitting
   - Prevents the model from creating leaves with very few samples

### Cross-Validation Explained

**5-Fold Cross-Validation**:

```
Training Data: [████████████████████]
Split into 5 folds:
  Fold 1: [████] [████] [████] [████] [████]
           ↑test  train  train  train  train
  Fold 2: [████] [████] [████] [████] [████]
           train  ↑test  train  train  train
  ... (repeat for all 5 folds)
```

Each fold gives a score, and we average them. This gives a more reliable estimate of how well the model will perform on new data.

---

## Visualizations

The script creates several plots to help understand the model:

### 1. Training History (`training_history.png`)

- Shows how the model's error (RMSE) decreases over time
- Two plots:
  - **Left**: Training RMSE vs Validation RMSE over iterations
  - **Right**: Shows the gap between training and validation (overfitting indicator)
- **What to look for**: Validation error should decrease and stay close to training error

### 2. Feature Importance (`feature_importance.png`)

- Horizontal bar chart of the top 20 most important features
- Shows which features the model relies on most for predictions
- **What to look for**: Features that make intuitive sense (e.g., "OverallQual", "GrLivArea")

### 3. Validation Predictions (`validation_predictions.png`)

- Two plots:
  - **Left**: Predicted vs Actual prices (scatter plot with perfect prediction line)
  - **Right**: Residuals plot (errors vs predictions)
- **What to look for**: Points should cluster around the diagonal line; residuals should be random (no patterns)

### 4. Training Predictions (`train_predictions.png`)

- Same as validation predictions, but for training data
- **What to look for**: Compare with validation - if training is much better, model might be overfitting

---

## Output Files

After training, the script saves several files:

### Model Files

- **`models/catboost_model.cbm`**: The trained CatBoost model (can be loaded for predictions)
- **`models/catboost_preprocessor.pkl`**: The preprocessing pipeline (needed to transform new data the same way)
- **`models/catboost_metadata.pkl`**: Metadata including:
  - Which columns are numeric vs categorical
  - Categorical feature indices
  - Best hyperparameters found
  - Performance metrics (RMSE, MAE, R²)

### Analysis Files

- **`models/catboost_feature_importance.csv`**: Feature importance scores for all features

### Visualization Files (in `models/plots/`)

- `training_history.png`: Training curves
- `feature_importance.png`: Top features
- `validation_predictions.png`: Validation set predictions
- `train_predictions.png`: Training set predictions

---

## Key Concepts Explained Simply

### Overfitting vs Underfitting

**Overfitting**: Model memorizes training data but fails on new data

- **Sign**: Training error much lower than validation error
- **Solution**: More regularization, simpler model, more data

**Underfitting**: Model is too simple to capture patterns

- **Sign**: Both training and validation errors are high
- **Solution**: More complex model, more features, less regularization

**Good Generalization**: Model performs similarly on training and validation

- **Sign**: Training and validation errors are close
- **Goal**: This is what we want!

### Early Stopping

Instead of training for all iterations, stop when:

- Validation error stops improving
- Prevents overfitting by stopping before the model memorizes training data
- Set to 100 rounds: if validation error doesn't improve for 100 iterations, stop

### Regularization

Techniques to prevent overfitting:

- **L2 regularization** (`l2_leaf_reg`): Penalizes large weights
- **Subsampling** (`subsample=0.8`): Uses random 80% of data for each tree
- **Tree depth limits**: Shallower trees are simpler
- **Minimum samples per leaf**: Prevents very specific rules

### RMSE, MAE, R² Explained

**RMSE (Root Mean Squared Error)**:

- Average prediction error, but larger errors are penalized more
- In dollars: If RMSE = $25,000, predictions are off by ~$25k on average
- Lower is better

**MAE (Mean Absolute Error)**:

- Average absolute prediction error
- In dollars: If MAE = $20,000, predictions are off by $20k on average
- Lower is better

**R² (R-squared, Coefficient of Determination)**:

- How much of the variance in prices the model explains
- Range: 0 to 1 (or negative if model is worse than just predicting the mean)
- 1.0 = perfect predictions, 0.8 = model explains 80% of variance
- Higher is better

---

## Running the Script

Simply run:

```bash
python train_catboost_model.py
```

The script will:

1. Check for GPU availability (uses GPU if available, falls back to CPU)
2. Load and preprocess the data
3. Perform grid search (this takes a while!)
4. Train the final model
5. Generate visualizations
6. Save all artifacts

**Expected runtime**:

- Grid search: 30-60 minutes (depending on hardware)
- Final model training: 1-5 minutes
- Total: ~1 hour on CPU, ~15-30 minutes on GPU

---

## Next Steps

After training, you can use the saved model to make predictions on new data using `predict_catboost.py` (if it exists) or by loading the model and preprocessor files.

---

## Summary

This script implements a complete machine learning pipeline:

1. **Data cleaning**: Removes useless columns
2. **Smart preprocessing**: Handles missing values, outliers, and skewed distributions
3. **Leverages CatBoost**: Uses its native categorical handling
4. **Hyperparameter optimization**: Finds the best model settings
5. **Robust training**: Uses cross-validation and early stopping
6. **Comprehensive evaluation**: Multiple metrics and visualizations
7. **Production-ready**: Saves everything needed for future predictions

The result is a well-tuned model that can accurately predict house prices while avoiding overfitting!
