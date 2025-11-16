## Column Types

- **Structural categorical**: `MSSubClass`, `MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `Utilities`, `LotConfig`, `LandSlope`, `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `Foundation`, `Heating`, `CentralAir`, `Electrical`, `Functional`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `PavedDrive`, `Fence`, `MiscFeature`, `SaleType`, `SaleCondition`.
- **Ordinal quality/condition**: `OverallQual`, `OverallCond`, `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`.
- **Temporal numeric**: `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `MoSold`, `YrSold`.
- **Size/area numeric**: `LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `MiscVal`.
- **Count features**: `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `Bedroom`, `Kitchen`, `TotRmsAbvGrd`, `Fireplaces`, `GarageCars`.
- **Binary categorical**: `CentralAir`.
- **Mixed categorical (quality + NA)**: `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `Fence`, `MiscFeature`.

## Likely Irrelevant Columns for SalePrice

- `Utilities`: nearly all entries are `AllPub`, so variance is minimal and predictive power is negligible.
- `Street`, `Alley`, `Condition2`, `RoofMatl`: highly skewed distributions where most homes fall into a single category; their contribution to price models is typically noise.
- `PoolQC`, `PoolArea`, `MiscFeature`, `MiscVal`: extremely sparse or zero-heavy columns; without special handling they act as outliers rather than meaningful predictors.
- `MoSold`, `YrSold`: Ames data spans a narrow timeframe with limited seasonality; unless modeling macro trends, these act as noise.
- `LotConfig`, `LandSlope`: historically show very weak correlation with SalePrice; safe to drop when simplifying feature sets unless further feature engineering is planned.

