from src.preprocess import load_forecast_data, load_survival_data, load_gdp_data
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
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

    # Sort and remove any duplicate years
    series = series.sort_values("ds").drop_duplicates("ds")

    return series


def add_gdp_regressor(series, country, gdp_df):
    """
    Merges GDP into the series as an external regressor for Prophet.
    Prophet can use additional variables beyond just time — this lets
    the model learn how GDP changes affect cancer rates.

    For future years, the last known GDP value is carried forward.
    """
    country_gdp = gdp_df[gdp_df["entity"] == country][["year", "gdp"]].copy()
    country_gdp["ds"] = pd.to_datetime(country_gdp["year"], format="%Y")
    country_gdp = country_gdp[["ds", "gdp"]]

    series = series.merge(country_gdp, on="ds", how="left")

    # Forward/backward fill gaps so Prophet doesn't get NaN regressors
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

            # Add GDP regressor if available for this country
            if use_gdp and gdp_df is not None:
                series  = add_gdp_regressor(series, country, gdp_df)
                has_gdp = ("gdp" in series.columns and
                           series["gdp"].notna().sum() > 2)
            else:
                has_gdp = False

            # Build model
            # yearly_seasonality=False — cancer data is annual,
            #   there's no within-year seasonality to learn
            # interval_width=0.95 — 95% confidence interval
            model = Prophet(yearly_seasonality=False, interval_width=0.95)

            if has_gdp:
                model.add_regressor("gdp")

            model.fit(series)

            # Build future dataframe
            future = model.make_future_dataframe(periods=periods, freq="YE")

            if has_gdp:
                # Extend GDP into future using last known value
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
    Separate from run_forecast() because:
      - different source file format (wide CSV, not GBD)
      - different value column (survival_rate, not rate)
      - predictions must be clamped to 0–100%

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


def plot_forecast(series, forecast, country, cancer_type,
                  ylabel="Death Rate per 100,000",
                  subfolder="forecast"):
    """
    Saves a forecast plot showing:
      - Actual data points (black dots)
      - Forecast line (blue)
      - 95% confidence interval (blue shading)
      - Vertical line marking where forecast begins
    """
    os.makedirs(f"outputs/{subfolder}", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Actual data
    ax.scatter(series["ds"], series["y"],
               color="black", label="Actual", zorder=5, s=40)

    # Forecast line
    ax.plot(forecast["ds"], forecast["yhat"],
            color="blue", label="Forecast")

    # Confidence interval
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2, color="blue", label="95% Confidence Interval"
    )

    # Mark where forecast starts
    last_actual = series["ds"].max()
    ax.axvline(last_actual, color="gray", linestyle="--",
               alpha=0.7, label="Forecast start")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{cancer_type} — {country}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    filename = (f"outputs/{subfolder}/{country}_{cancer_type}.png"
                .replace(" ", "_"))
    plt.savefig(filename)
    plt.close()
