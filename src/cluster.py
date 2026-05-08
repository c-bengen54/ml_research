from src.preprocess import load_clustering_data, scale_features
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

os.makedirs("outputs/cluster", exist_ok=True)


def get_optimal_clusters(X_scaled, k_max=10):
    """
    Runs elbow method AND silhouette scoring side by side.

    Elbow method  — look for where inertia stops dropping sharply.
    Silhouette    — higher score = better separated clusters (max 1.0).
                    This removes the guesswork of reading the elbow plot.

    Returns the k with the best silhouette score.
    """
    inertias    = []
    silhouettes = []
    k_range     = list(range(2, k_max + 1))

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # ── Interactive elbow + silhouette chart ─────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method — Pick the bend point",
                        "Silhouette Score — Higher is Better")
    )

    fig.add_trace(
        go.Scatter(
            x=k_range, y=inertias,
            mode="lines+markers",
            marker=dict(size=8),
            line=dict(color="#3b82f6"),
            name="Inertia",
            hovertemplate="k=%{x}<br>Inertia=%{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=k_range, y=silhouettes,
            mode="lines+markers",
            marker=dict(size=8, color="#22c55e"),
            line=dict(color="#22c55e"),
            name="Silhouette",
            hovertemplate="k=%{x}<br>Score=%{y:.3f}<extra></extra>"
        ),
        row=1, col=2
    )

    best_k = k_range[int(np.argmax(silhouettes))]

    # Highlight the best k
    fig.add_vline(x=best_k, line_dash="dash", line_color="red",
                  annotation_text=f"Best k={best_k}", row=1, col=2)

    fig.update_layout(
        title="Cluster Evaluation — Elbow & Silhouette",
        height=450,
        showlegend=False,
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Number of Clusters (k)")
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    fig.write_html("outputs/cluster/elbow_silhouette.html")
    print(f"Elbow + silhouette chart saved → outputs/cluster/elbow_silhouette.html")
    print(f"Best k by silhouette score: {best_k}")
    return best_k


def run_clustering(n_clusters=None):
    # ── 1. Load data ─────────────────────────────────────────────
    df = load_clustering_data()

    # ── 2. Scale ─────────────────────────────────────────────────
    X_scaled, scaler = scale_features(df)

    # ── 3. Find optimal k if not specified ───────────────────────
    if n_clusters is None:
        n_clusters = get_optimal_clusters(X_scaled)
    else:
        get_optimal_clusters(X_scaled)

    # ── 4. Train KMeans ──────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette score for k={n_clusters}: {score:.3f}")
    print("  > 0.50 = strong clusters")
    print("  0.25 - 0.50 = reasonable clusters")
    print("  < 0.25 = weak clusters, consider different k")

    # ── 5. PCA for visualization ─────────────────────────────────
    pca       = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"\nPCA variance explained (3 components): "
          f"{pca.explained_variance_ratio_.sum():.2%}")

    # ── 6. Attach results to DataFrame ───────────────────────────
    df["cluster"] = labels
    df["pc1"]     = X_reduced[:, 0]
    df["pc2"]     = X_reduced[:, 1]
    df["pc3"]     = X_reduced[:, 2]

    # ── 7. Output ─────────────────────────────────────────────────
    plot_clusters(df)
    plot_gdp_vs_cluster(df)
    explain_pca(pca, df)
    summarize_clusters(df)

    return df


def plot_clusters(df):
    """
    Interactive 2D PCA scatter plot. Hover to see country name,
    cluster, and GDP. Click legend to show/hide clusters.
    """
    colors = px.colors.qualitative.Set2
    fig    = go.Figure()

    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id].copy()
        subset.index.name = "country"
        subset = subset.reset_index()

        gdp_text = (
            subset["gdp"].apply(lambda v: f"${v/1e9:.1f}B" if pd.notna(v) else "N/A")
            if "gdp" in subset.columns else ["N/A"] * len(subset)
        )

        fig.add_trace(go.Scatter(
            x=subset["pc1"],
            y=subset["pc2"],
            mode="markers+text",
            name=f"Cluster {cluster_id} (n={len(subset)})",
            marker=dict(
                size=10,
                color=colors[cluster_id % len(colors)],
                opacity=0.8,
                line=dict(width=1, color="white")
            ),
            text=subset["country"],
            textposition="top center",
            textfont=dict(size=8),
            customdata=np.column_stack([
                subset["country"],
                gdp_text,
                [cluster_id] * len(subset)
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[2]}<br>"
                "GDP (PPP): %{customdata[1]}<br>"
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Country Cancer Profile Clusters (PCA 2D)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Cluster",
        template="plotly_white",
        height=650,
        hovermode="closest"
    )

    fig.write_html("outputs/cluster/cluster_plot.html")
    print("Cluster plot saved → outputs/cluster/cluster_plot.html")


def plot_gdp_vs_cluster(df):
    """
    Interactive boxplot of GDP distribution per cluster.
    Hover to see median, quartiles, and outlier countries.
    """
    if "gdp" not in df.columns:
        print("GDP column not found — skipping GDP boxplot")
        return

    df_plot = df.reset_index()
    # The index is named "entity" from load_clustering_data()
    country_col = "entity" if "entity" in df_plot.columns else df_plot.columns[0]
    df_plot["cluster_label"] = df_plot["cluster"].apply(lambda c: f"Cluster {c}")

    fig = px.box(
        df_plot,
        x="cluster_label",
        y="gdp",
        color="cluster_label",
        points="all",
        hover_name=country_col,
        hover_data={"gdp": ":,.0f", "cluster_label": False},
        labels={"gdp": "GDP PPP (current international $)",
                "cluster_label": "Cluster"},
        title="GDP Distribution by Cluster",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis_tickformat="$.2s"
    )

    fig.write_html("outputs/cluster/cluster_gdp_boxplot.html")
    print("GDP boxplot saved → outputs/cluster/cluster_gdp_boxplot.html")


def explain_pca(pca, df):
    """
    Prints which cancer types are the strongest drivers of PC1 and PC2.
    """
    feature_cols = [c for c in df.columns
                    if c not in ["cluster", "pc1", "pc2", "pc3"]]

    components = pd.DataFrame(
        pca.components_[:2],
        columns=feature_cols,
        index=["PC1", "PC2"]
    )

    print("\nTop drivers of PC1:")
    print(components.loc["PC1"].abs().sort_values(ascending=False).head(5).round(3))
    print("\nTop drivers of PC2:")
    print(components.loc["PC2"].abs().sort_values(ascending=False).head(5).round(3))


def summarize_clusters(df):
    """
    Prints mean value of every feature per cluster.
    """
    feature_cols = [c for c in df.columns
                    if c not in ["cluster", "pc1", "pc2", "pc3"]]

    summary = df.groupby("cluster")[feature_cols].mean().round(2)
    print("\nCluster Summary (mean values per cluster):")
    print(summary.to_string())
