from src.preprocess import load_gdp_data, load_gbd, load_tobacco_data, load_disease_burden
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import plotly.graph_objects as go
import os

os.makedirs("outputs/regression", exist_ok=True)

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
    print(f"Test  MAE: {mean_absolute_error(y_test,  test_pred):.3f}\n")
    print(f"Train R²:  {r2_score(y_train, train_pred):.3f}")
    print(f"Test  R²:  {r2_score(y_test,  test_pred):.3f}\n")

    # ── 6. Feature importance ─────────────────────────────────────
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importance.round(3))

    
    # ── 7. Decision Tree and Variance graphing ─────────────────────────────────────
    # Perfect prediction line
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())

    fig = go.Figure()

    # Scatter of predictions
    fig.add_trace(go.Scatter(
        x=y_test,
        y=test_pred,
        mode="markers",
        marker=dict(size=5, color="#3b82f6", opacity=0.5),
        name="Predictions",
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
    ))

    # Perfect prediction diagonal
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Perfect Prediction"
    ))

    fig.update_layout(
        title="Predicted vs Actual Death Rates",
        xaxis_title="Actual Death Rate (per 100,000)",
        yaxis_title="Predicted Death Rate (per 100,000)",
        template="plotly_white",
        height=600
    )

    fig.write_html("outputs/regression/predicted_vs_actual.html")
    print("Predicted vs actual plot saved → outputs/regression/predicted_vs_actual.html")

    tree_index = 0

    tree_df = model.get_booster().trees_to_dataframe()
    tree = tree_df[tree_df["Tree"] == tree_index].copy()

    # Build node positions using a simple top-down layout
    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []

    # Assign depth to each node
    tree["depth"] = 0
    for _, row in tree.iterrows():
        if row["Yes"] != "Leaf":
            yes_mask = tree["ID"] == row["Yes"]
            no_mask  = tree["ID"] == row["No"]
            tree.loc[yes_mask, "depth"] = row["depth"] + 1
            tree.loc[no_mask,  "depth"] = row["depth"] + 1

    max_depth = tree["depth"].max()

    # Assign x positions per depth level
    for depth in range(int(max_depth) + 1):
        level_nodes = tree[tree["depth"] == depth]
        n = len(level_nodes)
        for i, (idx, row) in enumerate(level_nodes.iterrows()):
            x = (i + 1) / (n + 1)
            y = 1 - (depth / (max_depth + 1))
            tree.at[idx, "x"] = x
            tree.at[idx, "y"] = y

    # Build edges
    for _, row in tree.iterrows():
        if row["Feature"] != "Leaf":
            for child_id in [row["Yes"], row["No"]]:
                child = tree[tree["ID"] == child_id]
                if not child.empty:
                    edge_x += [row["x"], child["x"].values[0], None]
                    edge_y += [row["y"], child["y"].values[0], None]

    # Build nodes
    for _, row in tree.iterrows():
        node_x.append(row["x"])
        node_y.append(row["y"])
        if row["Feature"] == "Leaf":
            node_text.append(f"Leaf<br>Value: {float(row['Gain']):.3f}")
            node_color.append("#22c55e")
        else:
            node_text.append(
                f"<b>{row['Feature']}</b><br>"
                f"Split: {float(row['Split']):.2f}<br>"
                f"Gain: {float(row['Gain']):.3f}"
            )
            node_color.append("#3b82f6")

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#94a3b8", width=1),
        hoverinfo="skip",
        showlegend=False
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=20, color=node_color,
                    line=dict(width=1, color="white")),
        text=[t.split("<br>")[0] for t in node_text],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False
    ))

    fig.update_layout(
        title=f"XGBoost Tree {tree_index} Structure",
        template="plotly_white",
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.write_html(f"outputs/regression/tree_{tree_index}.html")
    print(f"Tree diagram saved → outputs/regression/tree_{tree_index}.html")
    print("\nNote: open tree_0.html and zoom in to read individual node splits — the tree is deep and dense by design.")
    return model