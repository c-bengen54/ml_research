from src.cluster import run_clustering
from src.forecast import run_forecast, run_forecast_survival

# ── Clustering ────────────────────────────────────────────────────────────────
# Pass None to auto-detect best k via silhouette score
# Or pass a specific number: run_clustering(n_clusters=4)
df_clustered = run_clustering(n_clusters=2)


# ── Forecasting — all cancer death rates, all countries ───────────────────────
run_forecast(
    source="death_by_type",
    periods=20,
    use_gdp=True
)

# ── Forecasting — female cancers only ─────────────────────────────────────────
run_forecast(
    source="death_by_gender",
    filters={"sex": "Female"},
    cancer_types=["Breast cancer", "Cervical cancer", "Ovarian cancer"],
    periods=20,
    use_gdp=True
)

# ── Forecasting — male cancers only ─────────────────────────────────────────
run_forecast(
    source="death_by_gender",
    filters={"sex": "Male"},
    cancer_types=["Prostate cancer", "Lung cancer", "Colorectal cancer"],
    periods=20,
    use_gdp=True
)

# ── Forecasting — working age adults (20-54) ──────────────────────────────────
run_forecast(
    source="death_by_age_young",
    filters={"age_group": "20-54 years"},
    periods=20,
    use_gdp=True   # GDP regressor optional for age-specific slices
)

# ── Forecasting — retired age adults (50-84) ──────────────────────────────────
run_forecast(
    source="death_by_age_old",
    filters={"age_group": "50-84 years"},
    periods=20,
    use_gdp=True
)

# ── Forecasting — survival rates ──────────────────────────────────────────────
# Available cancer types: Colorectal, Ovarian, Stomach, Lung, Liver, Pancreatic
run_forecast_survival(periods=20)
