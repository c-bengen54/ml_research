# Cancer ML — Global Cancer Analysis & Forecasting

A machine learning project that discovers patterns in global cancer data, predicts cancer death rates, and forecasts future trends using clustering, regression, and time series models. Built in Python using scikit-learn, XGBoost, and Prophet.

---

## Project Structure

```
cancer-ml/
├── data/                                           # Raw CSV data files (see Data Sources below)
│   ├── cancer-death-rates-by-type.csv
│   ├── cancer-death-rate-by-gender-vs-type.csv
│   ├── cancer-death-rate-by-age-group-0-54.csv
│   ├── cancer-death-rate-by-age-group-50-84.csv
│   ├── tobacco-risks.csv
│   ├── diesease-burden-rate.csv
│   ├── five-year-survival-rates-by-cancer-type.csv
│   └── gdp.csv
├── src/
│   ├── preprocess.py       # Data loading, cleaning, and reshaping
│   ├── cluster.py          # KMeans clustering + PCA visualization
│   ├── regression.py       # XGBoost regression model
│   └── forecast.py         # Prophet time series forecasting
|
├── outputs/
│   ├── cluster/            # Cluster plots, elbow/silhouette charts
│   ├── forecast/           # Forecast plots per country and cancer type
│   │   └── survival/       # Survival rate forecast plots
│   └── regression/         # Regression variance and decision tree plots
├── main.py                 # Entry point — runs all models
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- A GitHub Codespace or local environment

### Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
scikit-learn
prophet
plotly
xgboost
numpy
```

### Using a Dev Container (recommended for Codespaces)

Add a `.devcontainer/devcontainer.json` file to auto-install on startup:

```json
{
  "name": "Cancer ML",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements.txt"
}
```

---

## Running the Project

```bash
# Run everything
python main.py

# Or run individual models from the terminal
python -c "from src.cluster import run_clustering; run_clustering()"
python -c "from src.regression import run_regression; run_regression()"
python -c "from src.forecast import run_forecast; run_forecast()"
```

All output plots are saved to the `outputs/` directory as interactive `.html` files. Open them directly in any browser by double-clicking, or in a Codespace via the Simple Browser (`Ctrl+Shift+P` → Simple Browser).

---

## How It Works

### `preprocess.py` — Data Layer
Central data pipeline. All other files import from here. Provides:

| Function | Returns | Used By |
|---|---|---|
| `load_gbd(key, filters)` | Standardized GBD DataFrame | All loaders |
| `load_clustering_data()` | Wide format: one row per country | `cluster.py` |
| `load_forecast_data(source, filters)` | Long format: one row per country/cancer/year | `forecast.py` |
| `load_survival_data()` | Long format survival rates | `forecast.py` |
| `load_age_data()` | Combined age group data | Optional use |
| `load_gdp_data()` | Long format GDP by country/year | Clustering + forecasting |
| `load_death_vs_gdp()` | Death rates merged with GDP | Analysis |
| `load_tobacco_data()` | Tobacco-attributed death % | `regression.py` |
| `load_disease_burden()` | DALY rates by cancer type | `regression.py` |
| `scale_features(X)` | Scaled numpy array + fitted scaler | `cluster.py` |

### `cluster.py` — Pattern Discovery
Uses **KMeans clustering** to group countries by their cancer burden profiles. Countries with similar death rates across cancer types are grouped together.

**Pipeline:**
1. Load wide-format country profiles (cancer rates + GDP)
2. Normalize with `StandardScaler` — prevents GDP from dominating distance calculations
3. Find optimal number of clusters using elbow method + silhouette scoring
4. Fit KMeans and assign cluster labels
5. Compress to 2D with PCA for visualization
6. Save interactive cluster plot, GDP boxplot, and cluster summary

**Outputs:**
- `outputs/cluster/elbow_silhouette.html` — interactive elbow and silhouette chart
- `outputs/cluster/cluster_plot.html` — interactive 2D scatter of country clusters
- `outputs/cluster/cluster_gdp_boxplot.html` — interactive GDP distribution per cluster
- Console: cluster summary table, PCA component drivers

**Usage:**
```python
from src.cluster import run_clustering

# Auto-detect best k
df_clustered = run_clustering()

# Or specify manually after checking the elbow plot
df_clustered = run_clustering(n_clusters=4)
```

### `regression.py` — Death Rate Prediction
Uses **XGBoost regression** to predict cancer death rates from GDP, tobacco attribution, disease burden, and cancer type. Achieves a test R² of 0.991 on the full dataset.

**Pipeline:**
1. Load and merge GBD death rates, GDP, tobacco attribution, and DALY burden data
2. Engineer features — mean country cancer rate and GDP rank per year
3. Encode cancer type as a numeric category
4. Split 80/20 into train and test sets
5. Optionally run GridSearchCV to find optimal hyperparameters
6. Evaluate train and test performance and print feature importance

**Feature importance (tuned model):**

| Feature | Importance |
|---|---|
| `cancer_type` | 0.713 |
| `mean_country_rate` | 0.169 |
| `gdp` | 0.066 |
| `gdp_rank` | 0.053 |
| `year` | 0.000 |

**Best parameters found for standard call with tuning:** `n_estimators=200`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.8`

**Usage:**
```python
from src.regression import run_regression

# Train on all cancer types with pre-tuned parameters (fast)
run_regression(tune=False)

# Train on all cancer types and re-run grid search (slow, ~5 min)
run_regression(tune=True)

# Train on a single cancer type, uses pre-tuned parameters
run_regression(cancer_type="Lung cancer", tune=False)
```

### `forecast.py` — Trend Forecasting
Uses **Facebook Prophet** to forecast cancer death rates and survival rates up to 10 years ahead. Optionally uses GDP as an external regressor to improve accuracy.

**Pipeline:**
1. Load long-format time series data
2. Filter to one country + one cancer type
3. Format for Prophet (`ds` and `y` columns)
4. Optionally attach GDP as a regressor
5. Fit Prophet model and generate forecast
6. Save interactive plot with actual data, forecast line, and 95% confidence interval

**Two forecast modes:**

`run_forecast()` — forecasts cancer **death rates** using GBD data. Supports filtering by sex or age group.

`run_forecast_survival()` — forecasts 5-year **survival rates** using ICBP data. Predictions are clamped to 0–100%.

**Outputs:** Interactive `.html` files per country/cancer combination with a range slider and 10y/20y/All time range buttons.

**Usage:**
```python
from src.forecast import run_forecast, run_forecast_survival

# All cancer types, all countries, with GDP regressor
run_forecast(source="death_by_type", periods=10, use_gdp=True)

# Female cancers only
run_forecast(
    source="death_by_gender",
    filters={"sex": "Female"},
    cancer_types=["Breast cancer", "Cervical cancer", "Ovarian cancer"]
)

# Working-age adults
run_forecast(
    source="death_by_age_young",
    filters={"age_group": "20-54 years"}
)

# Survival rate forecasts
run_forecast_survival(periods=10)
```

---

## Data Sources

### 1. Global Burden of Disease Study 2023 (GBD 2023)
**Files:** `cancer-death-rates-by-type.csv`, `cancer-death-rate-by-gender-vs-type.csv`, `cancer-death-rate-by-age-group-0-54.csv`, `cancer-death-rate-by-age-group-50-84.csv`, `tobacco-risks.csv`, `diesease-burden-rate.csv`

**Coverage:** 204 countries, 1990–2023, 34 cancer types

**Citation:**
> Global Burden of Disease Collaborative Network. *Global Burden of Disease Study 2023 (GBD 2023) Results.* Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. Available from https://vizhub.healthdata.org/gbd-results/

**License:** Free to use for non-commercial research and educational purposes with attribution. See [IHME Free-of-Charge Non-Commercial User Agreement](https://www.healthdata.org/data-tools-practices/data-practices/ihme-free-charge-non-commercial-user-agreement).

---

### 2. Five-Year Cancer Survival Rates (ICBP SURVMARK-2)
**File:** `five-year-survival-rates-by-cancer-type.csv`

**Coverage:** Selected countries, 1995–2014, cancer types: Colorectal, Ovarian, Stomach, Lung, Liver, Pancreatic

**Original source:** Global Cancer Observatory — ICBP SURVMARK-2 Cancer Survival Rates

**Citation:**
> Global Cancer Observatory (2019) – with minor processing by Our World in Data. "Five-year net survival rates by cancer type" [dataset]. Global Cancer Observatory, "ICBP SURVMARK-2 - Cancer Survival Rates" [original data]. Available from https://gco.iarc.fr/survival/survmark/

**Retrieved via:** Our World in Data — https://ourworldindata.org/grapher/five-year-survival-rates-by-cancer-type

**Note:** Net survival rates are age-standardized using International Cancer Survival Standard (ICSS) weights. Net survival estimates the probability of survival from cancer specifically, removing the effect of other causes of death.

---

### 3. World Bank GDP (PPP, Current International $)
**File:** `gdp.csv`

**Indicator:** `NY.GDP.MKTP.PP.CD` — GDP, PPP (current international $)

**Coverage:** 266 countries, 1960–2025

**Citation:**
> World Bank. *GDP, PPP (current international $)* [dataset]. World Development Indicators. Washington, D.C.: The World Bank Group. Available from https://data.worldbank.org/indicator/NY.GDP.MKTP.PP.CD

**License:** Creative Commons Attribution 4.0 (CC BY 4.0). See [World Bank Terms of Use](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets).

---

## Notes on Data

**Country name alignment:** GBD and World Bank use different naming conventions for 31 countries (e.g., "United States of America" vs "United States"). A mapping table in `preprocess.py` handles this automatically on load.

**Age file naming:** The files `cancer-death-rate-by-age-group-0-54.csv` and `cancer-death-rate-by-age-group-50-84.csv` are named for their intended age ranges but were downloaded with swapped contents from the GBD tool. The code in `preprocess.py` uses the correct file for each range — verify with `load_gbd("death_by_age_young")` and check the `age_group` column if results seem unexpected.

**Survival data limitations:** The survival file covers only 6 cancer types and ends in 2014. Forecasts from this file carry more uncertainty than death rate forecasts due to the shorter and older time series.

---

## Limitations

- Survival rate data ends in 2014 — forecasts beyond ~2025 should be interpreted cautiously
- GDP regressor uses last known value for future years — does not account for economic shocks or growth
- KMeans assumes spherical clusters — countries with unusual cancer profiles may not cluster well
- Small countries with sparse data are dropped during preprocessing (below 70% column coverage threshold)
- Regression model encodes cancer type as an arbitrary integer — meaningful only within the model, not across comparisons
- Regression feature importance shows `year` contributes nothing — death rates are stable enough year-over-year that the model gains no signal from it

---

## License

This project is for educational and research purposes. All data used remains subject to the licenses of the original data providers listed above.