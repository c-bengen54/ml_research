from src.preprocess import load_forecast_data, load_survival_data, load_gdp_data
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
import os

os.makedirs("outputs/forecast", exist_ok=True)


def prepare_series(df, country, cancer_type, value_col="rate"):
    """
    Filters to one country + cancer type and formats for Prophet.
    Prophet requires exactly two columns named 'ds' and 'y'.

    Args:
        df          : DataFrame from load_forecast_data() or load_survival_data()
        country     : country name string
        cancer_type : cancer type string
        value_col   : the column to use as y — 'rate' for death data,
                      'survival_rate' for survival data

    Returns:
        DataFrame with columns: ds (datetime), y (float)
    """
    mask = (
        (df["entity"] == country) &
        (df["cancer_type"] == cancer_type)
    )
    series = df[mask][["year", value_col]].copy()
    series.columns = ["ds", "y"]
    series["ds"] = pd.to_datetime(series["ds"], format="%Y")

    series = series.sort_values("ds").drop_duplicates("ds")
    return series


def add_gdp_regressor(series, country, gdp_df):
    """
    Merges GDP into the series as an external regressor for Prophet.
    For future years, the last known GDP value is carried forward.
    """
    country_gdp = gdp_df[gdp_df["entity"] == country][["year", "gdp"]].copy()
    country_gdp["ds"] = pd.to_datetime(country_gdp["year"], format="%Y")
    country_gdp = country_gdp[["ds", "gdp"]]

    series = series.merge(country_gdp, on="ds", how="left")
    series["gdp"] = series["gdp"].ffill().bfill()
    return series


def run_forecast(source="death_by_type", filters=None,
                 periods=10, use_gdp=True,
                 countries=None, cancer_types=None):
    """
    Forecasts cancer death rates for all country/cancer combinations.

    Args:
        source       : which dataset to forecast — any key from GBD_FILES in
                       preprocess.py (default: "death_by_type")
        filters      : dict to filter the source data before forecasting, e.g.
                       {"sex": "Female"} or {"age_group": "20-54 years"}
        periods      : how many years ahead to forecast (default: 10)
        use_gdp      : whether to include GDP as an external regressor
        countries    : list of country names to limit forecast, or None for all
        cancer_types : list of cancer types to limit forecast, or None for all

    Returns:
        List of dicts with keys: country, cancer_type, forecast, series
    """
    df     = load_forecast_data(source=source, filters=filters)
    gdp_df = load_gdp_data() if use_gdp else None

    if countries:
        df = df[df["entity"].isin(countries)]
    if cancer_types:
        df = df[df["cancer_type"].isin(cancer_types)]

    results = []

    for cancer in df["cancer_type"].unique():
        for country in df["entity"].unique():

            series = prepare_series(df, country, cancer, value_col="rate")
            if len(series) < 3:
                continue

            if use_gdp and gdp_df is not None:
                series  = add_gdp_regressor(series, country, gdp_df)
                has_gdp = ("gdp" in series.columns and
                           series["gdp"].notna().sum() > 2)
            else:
                has_gdp = False

            model = Prophet(yearly_seasonality=False, interval_width=0.95)

            if has_gdp:
                model.add_regressor("gdp")

            model.fit(series)

            future = model.make_future_dataframe(periods=periods, freq="YE")

            if has_gdp:
                last_gdp = series["gdp"].iloc[-1]
                future   = future.merge(series[["ds", "gdp"]], on="ds", how="left")
                future["gdp"] = future["gdp"].fillna(last_gdp)

            forecast = model.predict(future)

            results.append({
                "country":     country,
                "cancer_type": cancer,
                "forecast":    forecast,
                "series":      series,
            })

            plot_forecast(series, forecast, country, cancer,
                          ylabel="Death Rate per 100,000")

    print(f"\nForecasts complete — {len(results)} series processed")
    return results


def run_forecast_survival(periods=10, countries=None, cancer_types=None):
    """
    Forecasts 5-year survival rates using the Our World in Data survival file.

    Cancer types available: Colorectal, Ovarian, Stomach, Lung, Liver, Pancreatic
    Year range: 1995–2014 (limited — forecasts carry more uncertainty)
    """
    df = load_survival_data()

    if countries:
        df = df[df["entity"].isin(countries)]
    if cancer_types:
        df = df[df["cancer_type"].isin(cancer_types)]

    results = []

    for cancer in df["cancer_type"].unique():
        for country in df["entity"].unique():

            series = prepare_series(df, country, cancer,
                                    value_col="survival_rate")

            if len(series) < 3:
                continue

            model = Prophet(yearly_seasonality=False, interval_width=0.95)
            model.fit(series)

            future   = model.make_future_dataframe(periods=periods, freq="YE")
            forecast = model.predict(future)

            # Survival rate must stay within 0–100%
            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                forecast[col] = forecast[col].clip(0, 100)

            results.append({
                "country":     country,
                "cancer_type": cancer,
                "forecast":    forecast,
                "series":      series,
            })

            plot_forecast(series, forecast, country, cancer,
                          ylabel="5-Year Survival Rate (%)",
                          subfolder="forecast/survival")

    print(f"\nSurvival forecasts complete — {len(results)} series processed")
    return results


def plot_forecast(series, forecast, country, cancer_type, ylabel="Death Rate per 100,000", subfolder="forecast"):
    """
    Saves an interactive Plotly forecast chart as an HTML file showing:
      - Actual data points (black dots, hoverable)
      - Forecast line (blue)
      - 95% confidence interval (shaded blue)
      - Vertical line marking where forecast begins
      - Toggle buttons to show/hide confidence interval
    """
    os.makedirs(f"outputs/{subfolder}", exist_ok=True)

    last_actual = series["ds"].max()

    # Split forecast into historical fit and future projection
    hist_fc  = forecast[forecast["ds"] <= last_actual]
    future_fc = forecast[forecast["ds"] > last_actual]

    fig = go.Figure()

    # ── Confidence interval (future only) ────────────────────────
    fig.add_trace(go.Scatter(
        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval",
        hoverinfo="skip"
    ))

    # ── Historical confidence interval (lighter) ──────────────────
    fig.add_trace(go.Scatter(
        x=pd.concat([hist_fc["ds"], hist_fc["ds"][::-1]]),
        y=pd.concat([hist_fc["yhat_upper"], hist_fc["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.07)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Historical CI",
        hoverinfo="skip",
        showlegend=False
    ))

    # ── Forecast line (full range) ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        line=dict(color="#3b82f6", width=2),
        name="Forecast",
        hovertemplate="<b>%{x|%Y}</b><br>Forecast: %{y:.2f}<extra></extra>"
    ))

    # ── Actual data points ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=series["ds"],
        y=series["y"],
        mode="markers",
        marker=dict(color="black", size=7, symbol="circle",
                    line=dict(width=1, color="white")),
        name="Actual",
        hovertemplate="<b>%{x|%Y}</b><br>Actual: %{y:.2f}<extra></extra>"
    ))

    # ── Vertical line at forecast start ──────────────────────────
    fig.add_vline(
        x=last_actual.timestamp() * 1000,  # Plotly uses ms for datetime x-axes
        line_dash="dash",
        line_color="gray",
        line_width=1.5,
        annotation_text="Forecast start",
        annotation_position="top right",
        annotation_font_size=11
    )

    fig.update_layout(
        title=dict(
            text=f"{cancer_type} — {country}",
            font=dict(size=16)
        ),
        xaxis_title="Year",
        yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )

    fig.update_xaxes(
        rangeslider=dict(visible=True),   # scroll to zoom into time range
        rangeselector=dict(
            buttons=[
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(count=20, label="20y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]
        )
    )

    safe_name = f"{country}_{cancer_type}".replace(" ", "_")
    filepath  = f"outputs/{subfolder}/{safe_name}.html"
    fig.write_html(filepath)
