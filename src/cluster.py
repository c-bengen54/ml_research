from src.preprocess import load_clustering_data, scale_features
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    inertias     = []
    silhouettes  = []
    k_range      = range(2, k_max + 1)

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, marker="o")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method — Pick the bend point")

    ax2.plot(k_range, silhouettes, marker="o", color="green")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score — Higher is Better")

    plt.tight_layout()
    plt.savefig("outputs/cluster/elbow_silhouette.png")
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"Elbow + silhouette plot saved → outputs/cluster/elbow_silhouette.png")
    print(f"Best k by silhouette score: {best_k}")
    return best_k


def run_clustering(n_clusters=None):
    # ── 1. Load data ─────────────────────────────────────────────
    # GDP is now included inside load_clustering_data()
    df = load_clustering_data()

    # ── 2. Scale ─────────────────────────────────────────────────
    X_scaled, scaler = scale_features(df)

    # ── 3. Find optimal k if not specified ───────────────────────
    if n_clusters is None:
        n_clusters = get_optimal_clusters(X_scaled)
    else:
        # Still run the plot even if k is manually set
        get_optimal_clusters(X_scaled)

    # ── 4. Train KMeans ──────────────────────────────────────────
    # n_init=10: runs 10 times with different random seeds,
    # keeps the best result — reduces chance of poor convergence
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette score for k={n_clusters}: {score:.3f}")
    print("  > 0.50 = strong clusters")
    print("  0.25 - 0.50 = reasonable clusters")
    print("  < 0.25 = weak clusters, consider different k")

    # ── 5. PCA for visualization ─────────────────────────────────
    # 3 components — plot 2D but keep a 3rd for extra variance
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
    fig, ax = plt.subplots(figsize=(12, 8))
    colors  = cm.tab10(np.linspace(0, 1, df["cluster"].nunique()))

    for cluster_id, color in zip(sorted(df["cluster"].unique()), colors):
        subset = df[df["cluster"] == cluster_id]
        ax.scatter(
            subset["pc1"], subset["pc2"],
            label=f"Cluster {cluster_id} (n={len(subset)})",
            color=color, alpha=0.7, s=80
        )
        # Annotate a sample of country names per cluster
        for _, row in subset.sample(min(3, len(subset)),
                                    random_state=42).iterrows():
            ax.annotate(row.name, (row["pc1"], row["pc2"]),
                        fontsize=7, alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Country Cancer Profile Clusters")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/cluster/cluster_plot.png")
    plt.close()
    print("Cluster plot saved → outputs/cluster/cluster_plot.png")


def plot_gdp_vs_cluster(df):
    """
    Boxplot of GDP distribution per cluster.
    Shows whether clusters separate by economic development.
    """
    if "gdp" not in df.columns:
        print("GDP column not found — skipping GDP boxplot")
        return

    fig, ax      = plt.subplots(figsize=(10, 6))
    clusters     = sorted(df["cluster"].unique())
    gdp_by_cluster = [df[df["cluster"] == c]["gdp"].dropna().values
                      for c in clusters]

    ax.boxplot(gdp_by_cluster, labels=[f"Cluster {c}" for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("GDP PPP (current international $)")
    ax.set_title("GDP Distribution by Cluster")
    plt.tight_layout()
    plt.savefig("outputs/cluster/cluster_gdp_boxplot.png")
    plt.close()
    print("GDP boxplot saved → outputs/cluster/cluster_gdp_boxplot.png")


def explain_pca(pca, df):
    """
    Prints which cancer types are the strongest drivers of PC1 and PC2.
    This tells you what the axes on your cluster plot actually mean.
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
    This is where the real insight is — what makes each cluster distinct.
    """
    feature_cols = [c for c in df.columns
                    if c not in ["cluster", "pc1", "pc2", "pc3"]]

    summary = df.groupby("cluster")[feature_cols].mean().round(2)
    print("\nCluster Summary (mean values per cluster):")
    print(summary.to_string())
