from src.preprocess import load_gdp_data, load_gbd, load_tobacco_data, load_disease_burden
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


def run_regression(cancer_type=None, tune=True):
    """
    Full regression pipeline predicting cancer death rates from
    GDP, tobacco, disease burden, and cancer type features.

    Loads and merges all data sources, engineers features, trains
    an XGBoost model, and prints evaluation metrics and feature
    importance.

    Args:
        cancer_type : optional string to filter to a single cancer type,
                      e.g. "Lung cancer". Pass None to train on all types.
        tune        : if True, runs GridSearchCV to find optimal hyperparameters.
                      if False, uses the pre-tuned parameters
                      (n_estimators=200, max_depth=8, learning_rate=0.1).
                      Set to False for faster runs after tuning once.

    Returns:
        Fitted XGBRegressor model with the best parameters found, else if tuning is false then uses pre-tuned parameters.

    Examples:
        run_regression()
            trains on all cancer types with grid search tuning

        run_regression(cancer_type="Lung cancer", tune=False)
            trains on lung cancer only using pre-tuned parameters
    """

    # ── 1. Load data ─────────────────────────────────────────────
    gbd_data = load_gbd("death_by_type")
    gdp_data = load_gdp_data()
    tobacco  = load_tobacco_data()[["entity", "year", "value"]].rename(columns={"value": "tobacco_pct"})
    burden   = load_disease_burden()[["entity", "year", "value"]].rename(columns={"value": "daly_rate"})

    if cancer_type:
        gbd_data = gbd_data[gbd_data["cancer_type"] == cancer_type]

    country_mean = (gbd_data.groupby(["entity", "year"])["value"]
                             .mean()
                             .reset_index()
                             .rename(columns={"value": "mean_country_rate"}))

    dataset = pd.merge(gbd_data, gdp_data, on=["entity", "year"])
    dataset = pd.merge(dataset, country_mean, on=["entity", "year"])
    dataset = pd.merge(dataset, tobacco, on=["entity", "year"], how="left")
    dataset = pd.merge(dataset, burden, on=["entity", "year"], how="left")
    dataset["gdp_rank"] = dataset.groupby("year")["gdp"].rank(ascending=False)

    # ── 2. Prepare features ───────────────────────────────────────
    X = dataset[["year", "gdp", "gdp_rank", "cancer_type", "mean_country_rate"]].copy()
    y = dataset["value"]
    X["cancer_type"] = X["cancer_type"].astype("category").cat.codes

    # ── 3. Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 4. Train XGBoost ──────────────────────────────────────────
    if tune:
        print("Tuning XGBoost...")
        param_grid = {
            "n_estimators":  [100, 200],
            "max_depth":     [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample":     [0.8, 1.0],
        }
        grid = GridSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            param_grid, cv=3, scoring="r2", verbose=2
        )
        grid.fit(X_train, y_train)
        print(f"Best parameters: {grid.best_params_}")
        model = grid.best_estimator_
    else:
        model = XGBRegressor(
            n_estimators=200, max_depth=8,
            learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    print(f"\nTrain MAE: {mean_absolute_error(y_train, train_pred):.3f}")
    print(f"Test  MAE: {mean_absolute_error(y_test,  test_pred):.3f}")
    print(f"Train R²:  {r2_score(y_train, train_pred):.3f}")
    print(f"Test  R²:  {r2_score(y_test,  test_pred):.3f}")

    # ── 6. Feature importance ─────────────────────────────────────
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importance.round(3))

    return model