import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── File paths ──────────────────────────────────────────────────────────────
GBD_FILES = {
    "death_by_type":      "data/cancer-death-rates-by-type.csv",
    "death_by_gender":    "data/cancer-death-rate-by-gender-vs-type.csv",
    "death_by_age_young": "data/cancer-death-rate-by-age-group-0-54.csv",   # actual ages: 0-54
    "death_by_age_old":   "data/cancer-death-rate-by-age-group-50-84.csv",  # actual ages: 50-84
    "tobacco":            "data/tobacco-risks.csv",                          # measure: Deaths %, has rei_name
    "disease_burden":     "data/diesease-burden-rate.csv",                   # measure: DALYs rate
}

SURVIVAL_FILE = "data/five-year-survival-rates-by-cancer-type.csv"
GDP_FILE      = "data/gdp.csv"

# ── Country name mapping (GBD → World Bank) ─────────────────────────────────
# GBD and World Bank use different names for the same countries.
# Keys are World Bank names, values are GBD names — we flip this on load
# so everything aligns to GBD naming throughout the pipeline.
COUNTRY_NAME_MAP = {
    "United States of America":              "United States",
    "Republic of Korea":                     "Korea, Rep.",
    "Democratic Republic of the Congo":      "Congo, Dem. Rep.",
    "Congo":                                 "Congo, Rep.",
    "Iran (Islamic Republic of)":            "Iran, Islamic Rep.",
    "Bolivia (Plurinational State of)":      "Bolivia",
    "Venezuela (Bolivarian Republic of)":    "Venezuela, RB",
    "United Republic of Tanzania":           "Tanzania",
    "Republic of Moldova":                   "Moldova",
    "Lao People's Democratic Republic":      "Lao PDR",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Côte d'Ivoire":                         "Cote d'Ivoire",
    "Türkiye":                               "Turkiye",
    "Egypt":                                 "Egypt, Arab Rep.",
    "Gambia":                                "Gambia, The",
    "Yemen":                                 "Yemen, Rep.",
    "Slovakia":                              "Slovak Republic",
    "Kyrgyzstan":                            "Kyrgyz Republic",
    "Micronesia (Federated States of)":      "Micronesia, Fed. Sts.",
    "Bahamas":                               "Bahamas, The",
    "Saint Kitts and Nevis":                 "St. Kitts and Nevis",
    "Saint Lucia":                           "St. Lucia",
    "Saint Vincent and the Grenadines":      "St. Vincent and the Grenadines",
    # Small territories with no World Bank data — left unmapped:
    # Cook Islands, Niue, Tokelau, Palestine, Puerto Rico,
    # Taiwan, United States Virgin Islands
}


# ── Shared GBD loader ────────────────────────────────────────────────────────
def load_gbd(key, filters=None):
    """
    Generic loader for any GBD file. Standardizes column names
    and optionally filters by any column value.

    Args:
        key     : one of the keys in GBD_FILES above
        filters : dict of {column: value} to filter rows, e.g.
                  {"sex": "Female"} or {"age_group": "20-54 years"}

    Returns:
        Cleaned DataFrame with standardized column names.
    """
    df = pd.read_csv(GBD_FILES[key])

    # Rename to consistent names across all GBD files
    df = df.rename(columns={
        "location_name": "entity",
        "cause_name":    "cancer_type",
        "year_start":    "year",
        "val":           "value",
        "upper":         "value_upper",
        "lower":         "value_lower",
        "sex_name":      "sex",
        "age_name":      "age_group",
        "metric_name":   "metric",
        "measure_name":  "measure",
        "rei_name":      "risk_factor",   # only present in tobacco file
    })

    # Drop columns not useful for ML
    df = df.drop(columns=[
        "population_group_id", "population_group_name",
        "measure_id", "location_id", "sex_id",
        "age_id", "cause_id", "metric_id",
        "rei_id", "year_end",
    ], errors="ignore")  # errors="ignore" safely skips absent columns

    # Apply filters if provided
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]

    return df


# ── Clustering loader ────────────────────────────────────────────────────────
def load_clustering_data():
    """
    Returns one row per country with each cancer type as its own column,
    enriched with GDP. Used by cluster.py.

    Shape: (n_countries, n_cancer_types + 1 gdp column)
    """
    df = load_gbd("death_by_type")

    # Most recent year per country/cancer — avoids duplicates in pivot
    df = (df.sort_values("year")
            .groupby(["entity", "cancer_type"])
            .last()
            .reset_index())

    # Wide format — one row per country, one column per cancer type
    wide = df.pivot(index="entity", columns="cancer_type", values="value")
    wide.columns.name = None  # remove the "cancer_type" axis label

    # Keep countries with at least 70% of cancer columns filled
    wide = wide.dropna(thresh=int(wide.shape[1] * 0.7))
    wide = wide.fillna(wide.mean())

    # Enrich with GDP
    gdp = load_gdp_data()
    gdp_latest = (gdp.sort_values("year")
                     .groupby("entity")["gdp"]
                     .last())
    wide = wide.join(gdp_latest, how="left")
    wide["gdp"] = wide["gdp"].fillna(wide["gdp"].mean())

    return wide


# ── Forecasting loader ───────────────────────────────────────────────────────
def load_forecast_data(source="death_by_type", filters=None):
    """
    Returns long-format time series data ready for Prophet.
    Used by forecast.py.

    Args:
        source  : GBD_FILES key — which dataset to forecast from
        filters : optional dict to filter rows, e.g. {"sex": "Female"}

    Returns:
        DataFrame with columns: entity, cancer_type, year, rate

    Examples:
        load_forecast_data()
            → all-cause death rates, all countries

        load_forecast_data("death_by_gender", filters={"sex": "Female"})
            → female death rates only

        load_forecast_data("death_by_age_young", filters={"age_group": "20-54 years"})
            → death rates for 20-54 age group
    """
    df = load_gbd(source, filters=filters)
    df = df[["entity", "cancer_type", "year", "value"]].dropna()
    df = df.rename(columns={"value": "rate"})
    return df


# ── Survival loader ──────────────────────────────────────────────────────────
def load_survival_data():
    """
    Loads Our World in Data survival file and returns long format.
    Used by forecast.py for survival rate forecasting.

    Columns: entity, code, year, cancer_type, survival_rate
    Cancer types: Colorectal, Ovarian, Stomach, Lung, Liver, Pancreatic
    Year range: 1995-2014
    """
    df = pd.read_csv(SURVIVAL_FILE)

    df_long = df.melt(
        id_vars=["Entity", "Code", "Year"],
        var_name="cancer_type",
        value_name="survival_rate"
    )
    df_long.columns = ["entity", "code", "year", "cancer_type", "survival_rate"]
    df_long = df_long.dropna()

    return df_long


# ── Age data loader ──────────────────────────────────────────────────────────
def load_age_data():
    """
    Combines both age group files into one DataFrame.
    Handles overlapping age brackets (50-69 and 65-74 appear in both files).

    Age groups covered: 0-14, 15-19, 20-54, 50-69, 65-74, 75-84
    """
    young = load_gbd("death_by_age_young")  # 0-14, 15-19, 20-54
    old   = load_gbd("death_by_age_old")    # 50-69, 65-74, 75-84

    combined = pd.concat([young, old])

    # Remove exact duplicates from overlapping age groups
    combined = combined.drop_duplicates(
        subset=["entity", "cancer_type", "year", "age_group"]
    )
    combined = combined.dropna(subset=["value"])

    return combined


# ── GDP loader ───────────────────────────────────────────────────────────────
def load_gdp_data():
    """
    Loads World Bank GDP PPP data in long format.
    Applies country name mapping so entity names align with GBD.

    Columns: entity, year, gdp
    Coverage: 204+ countries, 1960-2023
    """
    df = pd.read_csv(GDP_FILE, skiprows=4)

    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[["Country Name"] + year_cols]

    df = df.melt(
        id_vars=["Country Name"],
        var_name="year",
        value_name="gdp"
    )
    df.columns = ["entity", "year", "gdp"]
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=["gdp"])

    # Flip map: World Bank name → GBD name so everything uses GBD naming
    df["entity"] = df["entity"].replace(
        {v: k for k, v in COUNTRY_NAME_MAP.items()}
    )

    return df


# ── Death rate vs GDP loader ─────────────────────────────────────────────────
def load_death_vs_gdp():
    """
    Merges GBD cancer death rates with World Bank GDP by country + year.
    Replaces the old death-rate-from-cancers-vs-average-income.csv.

    Columns: entity, cancer_type, year, value, gdp (+ confidence intervals)
    """
    deaths = load_gbd("death_by_type")
    gdp    = load_gdp_data()
    return deaths.merge(gdp, on=["entity", "year"], how="inner")


# ── Tobacco attribution loader ───────────────────────────────────────────────
def load_tobacco_data():
    """
    Returns tobacco-attributable cancer death percentage by country/year.
    Metric is Percent — share of cancer deaths attributed to tobacco.
    """
    return load_gbd("tobacco")


# ── Disease burden (DALYs) loader ────────────────────────────────────────────
def load_disease_burden():
    """
    Returns DALY rates by cancer type, country, and year.
    DALYs capture both years of life lost and years lived with disability —
    more comprehensive than death rates alone.
    """
    return load_gbd("disease_burden")


# ── Shared scaler ────────────────────────────────────────────────────────────
def scale_features(X):
    """
    Applies StandardScaler to feature matrix X.
    Always fit on training data only — never on test data.

    Returns:
        X_scaled : numpy array
        scaler   : fitted scaler (save this to inverse_transform later)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
