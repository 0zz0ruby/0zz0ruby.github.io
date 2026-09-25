"""Simulation study for PCA + K-means customer segmentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "CC GENERAL.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"

RANDOM_STATE = 42

N_CLUSTERS = 4
SAMPLE_SIZES = [500, 1000, 3000]
NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0]
REPETITIONS = 20
REPRESENTATIVE_SAMPLE_SIZE = 1000
REPRESENTATIVE_NOISE_LEVEL = 1.0

SIMULATION_VARIABLES = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

AMOUNT_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]

COUNT_COLUMNS = ["CASH_ADVANCE_TRX", "PURCHASES_TRX"]
FREQUENCY_COLUMNS = [
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
]

CLUSTER_NAMES_CN = {
    0: "低活跃低消费客户",
    1: "高消费高信用额度客户",
    2: "取现依赖型客户",
    3: "高频分期消费客户",
}


def ensure_directories() -> None:
    """Create output folders used by this script."""
    for directory in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, SUMMARIES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def logistic(values: np.ndarray) -> np.ndarray:
    """Convert real-valued variables to the 0-1 interval."""
    return 1 / (1 + np.exp(-values))


def logit(values: np.ndarray) -> np.ndarray:
    """Convert proportions to logit scale with numerical clipping."""
    clipped = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def get_cluster_target_means() -> pd.DataFrame:
    """Define interpretable original-scale mean profiles for four customer types."""
    profiles = pd.DataFrame(
        [
            {
                "BALANCE": 650,
                "PURCHASES": 300,
                "ONEOFF_PURCHASES": 130,
                "INSTALLMENTS_PURCHASES": 170,
                "CASH_ADVANCE": 120,
                "PURCHASES_FREQUENCY": 0.28,
                "ONEOFF_PURCHASES_FREQUENCY": 0.10,
                "PURCHASES_INSTALLMENTS_FREQUENCY": 0.16,
                "CASH_ADVANCE_FREQUENCY": 0.04,
                "CASH_ADVANCE_TRX": 1.2,
                "PURCHASES_TRX": 5,
                "CREDIT_LIMIT": 2600,
                "PAYMENTS": 600,
                "MINIMUM_PAYMENTS": 300,
                "PRC_FULL_PAYMENT": 0.18,
                "TENURE": 11,
            },
            {
                "BALANCE": 2800,
                "PURCHASES": 6500,
                "ONEOFF_PURCHASES": 4200,
                "INSTALLMENTS_PURCHASES": 2300,
                "CASH_ADVANCE": 400,
                "PURCHASES_FREQUENCY": 0.92,
                "ONEOFF_PURCHASES_FREQUENCY": 0.55,
                "PURCHASES_INSTALLMENTS_FREQUENCY": 0.65,
                "CASH_ADVANCE_FREQUENCY": 0.04,
                "CASH_ADVANCE_TRX": 1.0,
                "PURCHASES_TRX": 70,
                "CREDIT_LIMIT": 11000,
                "PAYMENTS": 6500,
                "MINIMUM_PAYMENTS": 1500,
                "PRC_FULL_PAYMENT": 0.35,
                "TENURE": 12,
            },
            {
                "BALANCE": 4200,
                "PURCHASES": 500,
                "ONEOFF_PURCHASES": 250,
                "INSTALLMENTS_PURCHASES": 250,
                "CASH_ADVANCE": 4300,
                "PURCHASES_FREQUENCY": 0.25,
                "ONEOFF_PURCHASES_FREQUENCY": 0.12,
                "PURCHASES_INSTALLMENTS_FREQUENCY": 0.13,
                "CASH_ADVANCE_FREQUENCY": 0.55,
                "CASH_ADVANCE_TRX": 15,
                "PURCHASES_TRX": 8,
                "CREDIT_LIMIT": 7000,
                "PAYMENTS": 3400,
                "MINIMUM_PAYMENTS": 1900,
                "PRC_FULL_PAYMENT": 0.04,
                "TENURE": 11,
            },
            {
                "BALANCE": 1500,
                "PURCHASES": 2500,
                "ONEOFF_PURCHASES": 400,
                "INSTALLMENTS_PURCHASES": 2100,
                "CASH_ADVANCE": 250,
                "PURCHASES_FREQUENCY": 0.95,
                "ONEOFF_PURCHASES_FREQUENCY": 0.12,
                "PURCHASES_INSTALLMENTS_FREQUENCY": 0.90,
                "CASH_ADVANCE_FREQUENCY": 0.03,
                "CASH_ADVANCE_TRX": 1.0,
                "PURCHASES_TRX": 55,
                "CREDIT_LIMIT": 5500,
                "PAYMENTS": 2300,
                "MINIMUM_PAYMENTS": 800,
                "PRC_FULL_PAYMENT": 0.25,
                "TENURE": 12,
            },
        ],
        index=pd.Index(range(N_CLUSTERS), name="true_cluster"),
    )
    return profiles[SIMULATION_VARIABLES]


def transform_means_to_model_scale(original_means: pd.DataFrame) -> pd.DataFrame:
    """Move skewed and bounded variables to a scale suitable for normal noise."""
    transformed = original_means.copy()
    transformed[AMOUNT_COLUMNS + COUNT_COLUMNS] = np.log1p(
        transformed[AMOUNT_COLUMNS + COUNT_COLUMNS]
    )
    transformed[FREQUENCY_COLUMNS] = logit(transformed[FREQUENCY_COLUMNS].to_numpy())
    return transformed


def build_correlated_covariance() -> np.ndarray:
    """Create a positive-definite covariance matrix with interpretable correlations."""
    loadings = pd.DataFrame(
        {
            "purchase": [0.10, 0.85, 0.75, 0.50, 0.00, 0.70, 0.65, 0.45, 0.00, 0.00, 0.80, 0.25, 0.55, 0.15, 0.10, 0.00],
            "cash": [0.35, 0.00, 0.00, 0.00, 0.90, -0.10, 0.00, 0.00, 0.80, 0.85, 0.00, 0.00, 0.25, 0.30, -0.30, 0.00],
            "credit_payment": [0.35, 0.20, 0.20, 0.10, 0.25, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.80, 0.55, 0.45, 0.00, 0.05],
            "installment": [0.00, 0.30, -0.10, 0.75, 0.00, 0.35, -0.10, 0.75, 0.00, 0.00, 0.45, 0.10, 0.20, 0.00, 0.00, 0.00],
            "repayment": [-0.25, 0.10, 0.05, 0.05, -0.20, 0.25, 0.10, 0.10, -0.15, -0.15, 0.10, 0.15, 0.20, -0.30, 0.80, 0.10],
        },
        index=SIMULATION_VARIABLES,
    )
    variable_sd = pd.Series(
        {
            "BALANCE": 0.32,
            "PURCHASES": 0.36,
            "ONEOFF_PURCHASES": 0.40,
            "INSTALLMENTS_PURCHASES": 0.40,
            "CASH_ADVANCE": 0.42,
            "PURCHASES_FREQUENCY": 0.38,
            "ONEOFF_PURCHASES_FREQUENCY": 0.38,
            "PURCHASES_INSTALLMENTS_FREQUENCY": 0.38,
            "CASH_ADVANCE_FREQUENCY": 0.40,
            "CASH_ADVANCE_TRX": 0.36,
            "PURCHASES_TRX": 0.34,
            "CREDIT_LIMIT": 0.28,
            "PAYMENTS": 0.34,
            "MINIMUM_PAYMENTS": 0.35,
            "PRC_FULL_PAYMENT": 0.36,
            "TENURE": 0.20,
        }
    )

    factor_covariance = 0.12 * loadings.to_numpy() @ loadings.to_numpy().T
    idiosyncratic_covariance = np.diag(variable_sd.loc[SIMULATION_VARIABLES].to_numpy() ** 2)
    return factor_covariance + idiosyncratic_covariance


def allocate_cluster_counts(sample_size: int) -> np.ndarray:
    """Allocate observations to four true classes."""
    proportions = np.array([0.35, 0.20, 0.20, 0.25])
    counts = np.floor(sample_size * proportions).astype(int)
    counts[-1] += sample_size - counts.sum()
    return counts


def inverse_transform_simulated_values(model_scale_df: pd.DataFrame) -> pd.DataFrame:
    """Transform simulated values back to credit-card-like original scale."""
    simulated = model_scale_df.copy()
    simulated[AMOUNT_COLUMNS + COUNT_COLUMNS] = np.expm1(
        simulated[AMOUNT_COLUMNS + COUNT_COLUMNS]
    )
    simulated[FREQUENCY_COLUMNS] = logistic(simulated[FREQUENCY_COLUMNS].to_numpy())

    for column in AMOUNT_COLUMNS + COUNT_COLUMNS:
        simulated[column] = simulated[column].clip(lower=0)
    for column in FREQUENCY_COLUMNS:
        simulated[column] = simulated[column].clip(lower=0, upper=1)

    simulated["ONEOFF_PURCHASES"] = simulated["ONEOFF_PURCHASES"].clip(
        upper=simulated["PURCHASES"] * 1.2 + 1
    )
    simulated["INSTALLMENTS_PURCHASES"] = simulated["INSTALLMENTS_PURCHASES"].clip(
        upper=simulated["PURCHASES"] * 1.2 + 1
    )
    simulated["PURCHASES"] = np.maximum(
        simulated["PURCHASES"],
        0.75 * (simulated["ONEOFF_PURCHASES"] + simulated["INSTALLMENTS_PURCHASES"]),
    )
    simulated["CREDIT_LIMIT"] = np.maximum(
        simulated["CREDIT_LIMIT"], simulated["BALANCE"] * 1.05 + 500
    )
    simulated["PAYMENTS"] = np.maximum(simulated["PAYMENTS"], simulated["MINIMUM_PAYMENTS"] * 0.7)
    simulated["CASH_ADVANCE_TRX"] = np.rint(simulated["CASH_ADVANCE_TRX"]).clip(lower=0)
    simulated["PURCHASES_TRX"] = np.rint(simulated["PURCHASES_TRX"]).clip(lower=0)
    simulated["TENURE"] = np.rint(simulated["TENURE"]).clip(lower=6, upper=12)

    return simulated[SIMULATION_VARIABLES]


def generate_simulated_credit_card_data(
    sample_size: int,
    noise_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Generate a simulated credit-card dataset with known true clusters."""
    rng = np.random.default_rng(random_state)
    transformed_means = transform_means_to_model_scale(get_cluster_target_means())
    covariance = build_correlated_covariance() * (noise_level**2)
    counts = allocate_cluster_counts(sample_size)

    frames = []
    for cluster_id, count in enumerate(counts):
        simulated_values = rng.multivariate_normal(
            mean=transformed_means.loc[cluster_id].to_numpy(),
            cov=covariance,
            size=count,
        )
        cluster_df = pd.DataFrame(simulated_values, columns=SIMULATION_VARIABLES)
        cluster_df = inverse_transform_simulated_values(cluster_df)
        cluster_df["true_cluster"] = cluster_id
        cluster_df["true_cluster_name"] = CLUSTER_NAMES_CN[cluster_id]
        frames.append(cluster_df)

    simulated_df = pd.concat(frames, ignore_index=True)
    simulated_df = simulated_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return simulated_df


def perform_pca(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Standardize features and retain enough PCs to explain at least 80% variance."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    pca = PCA()
    scores_all = pca.fit_transform(scaled_features)
    explained_variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_))],
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    selected = np.where(explained_variance["cumulative_explained_variance_ratio"] >= 0.80)[0]
    n_components = int(selected[0] + 1) if len(selected) else feature_df.shape[1]
    n_components = max(2, n_components)

    pca_scores = pd.DataFrame(
        scores_all[:, :n_components],
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return pca_scores, explained_variance, n_components


def match_cluster_labels(true_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    """Map K-means labels to true labels using the Hungarian algorithm."""
    label_order = np.arange(N_CLUSTERS)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_order)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in predicted_labels])


def evaluate_clustering(
    pca_scores: pd.DataFrame,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Calculate external and internal clustering validation metrics."""
    matched_labels = match_cluster_labels(true_labels, predicted_labels)
    matched_accuracy = float(np.mean(matched_labels == true_labels))
    return {
        "ari": adjusted_rand_score(true_labels, predicted_labels),
        "nmi": normalized_mutual_info_score(true_labels, predicted_labels),
        "silhouette_score": silhouette_score(pca_scores, predicted_labels),
        "calinski_harabasz_index": calinski_harabasz_score(pca_scores, predicted_labels),
        "davies_bouldin_index": davies_bouldin_score(pca_scores, predicted_labels),
        "matched_accuracy": matched_accuracy,
    }


def run_single_simulation(
    sample_size: int,
    noise_level: float,
    repetition: int,
    random_state: int,
) -> dict[str, float]:
    """Run data generation, PCA, K-means, and metric calculation once."""
    simulated_df = generate_simulated_credit_card_data(
        sample_size=sample_size,
        noise_level=noise_level,
        random_state=random_state,
    )
    feature_df = simulated_df[SIMULATION_VARIABLES]
    pca_scores, explained_variance, n_components = perform_pca(feature_df)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state, n_init=10)
    predicted_labels = kmeans.fit_predict(pca_scores)
    metrics = evaluate_clustering(
        pca_scores=pca_scores,
        true_labels=simulated_df["true_cluster"].to_numpy(),
        predicted_labels=predicted_labels,
    )
    metrics.update(
        {
            "sample_size": sample_size,
            "noise_level": noise_level,
            "repetition": repetition,
            "n_clusters": N_CLUSTERS,
            "n_components": n_components,
            "cumulative_explained_variance": explained_variance.loc[
                n_components - 1, "cumulative_explained_variance_ratio"
            ],
        }
    )
    return metrics


def run_simulation_grid() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full sample-size by noise-level simulation grid."""
    design = pd.DataFrame(
        [
            {
                "sample_size": sample_size,
                "noise_level": noise_level,
                "repetitions": REPETITIONS,
                "n_clusters": N_CLUSTERS,
            }
            for sample_size in SAMPLE_SIZES
            for noise_level in NOISE_LEVELS
        ]
    )
    design.to_csv(TABLES_DIR / "simulation_design.csv", index=False)

    records = []
    for row in design.itertuples(index=False):
        for repetition in range(1, REPETITIONS + 1):
            seed = RANDOM_STATE + int(row.sample_size) * 10 + int(row.noise_level * 100) + repetition
            records.append(
                run_single_simulation(
                    sample_size=int(row.sample_size),
                    noise_level=float(row.noise_level),
                    repetition=repetition,
                    random_state=seed,
                )
            )

    raw_results = pd.DataFrame(records)
    raw_results = raw_results[
        [
            "sample_size",
            "noise_level",
            "repetition",
            "n_clusters",
            "n_components",
            "cumulative_explained_variance",
            "ari",
            "nmi",
            "silhouette_score",
            "calinski_harabasz_index",
            "davies_bouldin_index",
            "matched_accuracy",
        ]
    ]
    raw_results.to_csv(TABLES_DIR / "simulation_results_raw.csv", index=False)

    summary = summarize_simulation_results(raw_results)
    summary.to_csv(TABLES_DIR / "simulation_results_summary.csv", index=False)
    return design, raw_results, summary


def summarize_simulation_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation for each experimental condition."""
    metric_columns = [
        "n_components",
        "cumulative_explained_variance",
        "ari",
        "nmi",
        "silhouette_score",
        "calinski_harabasz_index",
        "davies_bouldin_index",
        "matched_accuracy",
    ]
    summary = (
        raw_results.groupby(["sample_size", "noise_level"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.insert(2, "repetitions", REPETITIONS)
    summary.insert(3, "n_clusters", N_CLUSTERS)
    return summary


def create_representative_outputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create the representative dataset, PCA table, profile table, and labels."""
    representative_df = generate_simulated_credit_card_data(
        sample_size=REPRESENTATIVE_SAMPLE_SIZE,
        noise_level=REPRESENTATIVE_NOISE_LEVEL,
        random_state=RANDOM_STATE,
    )
    representative_df.to_csv(
        TABLES_DIR / "simulation_single_dataset_preview.csv", index=False
    )

    feature_df = representative_df[SIMULATION_VARIABLES]
    pca_scores, explained_variance, n_components = perform_pca(feature_df)
    explained_variance.to_csv(
        TABLES_DIR / "simulation_pca_explained_variance.csv", index=False
    )

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    predicted_labels = kmeans.fit_predict(pca_scores)
    matched_labels = match_cluster_labels(
        representative_df["true_cluster"].to_numpy(), predicted_labels
    )

    profile = profile_true_clusters(representative_df)
    profile.to_csv(TABLES_DIR / "simulation_cluster_profile.csv", index=False)

    representative_scores = pca_scores[["PC1", "PC2"]].copy()
    representative_scores["true_cluster"] = representative_df["true_cluster"]
    representative_scores["predicted_cluster"] = predicted_labels
    representative_scores["matched_cluster"] = matched_labels
    representative_scores["n_components"] = n_components

    return representative_df, representative_scores, predicted_labels, matched_labels


def profile_true_clusters(simulated_df: pd.DataFrame) -> pd.DataFrame:
    """Create an original-scale profile table for the true simulated clusters."""
    profile = (
        simulated_df.groupby(["true_cluster", "true_cluster_name"])[SIMULATION_VARIABLES]
        .mean()
        .reset_index()
    )
    counts = (
        simulated_df["true_cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("true_cluster")
        .reset_index(name="count")
    )
    counts["percentage"] = counts["count"] / len(simulated_df)
    return counts.merge(profile, on="true_cluster", how="left")


def plot_pca_cluster_scatter(
    scores: pd.DataFrame,
    label_column: str,
    output_name: str,
    title: str,
    legend_title: str,
) -> None:
    """Plot PC1-PC2 scatter using true or K-means labels."""
    plot_df = scores.copy()
    plot_df[label_column] = plot_df[label_column].map(lambda value: f"Cluster {int(value)}")
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=label_column,
        palette="tab10",
        s=22,
        alpha=0.65,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_simulation_confusion_matrix(
    true_labels: np.ndarray,
    matched_labels: np.ndarray,
) -> None:
    """Plot the confusion matrix after label matching."""
    label_order = np.arange(N_CLUSTERS)
    matrix = confusion_matrix(true_labels, matched_labels, labels=label_order)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"True {label}" for label in label_order],
        columns=[f"Matched {label}" for label in label_order],
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=0.3)
    plt.xlabel("Matched K-means Cluster")
    plt.ylabel("True Cluster")
    plt.title("Simulation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "simulation_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_by_noise(
    summary: pd.DataFrame,
    metric_column: str,
    output_name: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot simulation metric means by noise level."""
    plt.figure(figsize=(9, 6))
    for sample_size in SAMPLE_SIZES:
        subset = summary.loc[summary["sample_size"] == sample_size].sort_values(
            "noise_level"
        )
        plt.plot(
            subset["noise_level"],
            subset[metric_column],
            marker="o",
            linewidth=1.8,
            label=f"n={sample_size}",
        )
    plt.xlabel("Noise Level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.05 if metric_column != "silhouette_score_mean" else None)
    plt.legend(title="Sample Size")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_by_sample_size(summary: pd.DataFrame) -> None:
    """Plot average ARI, NMI, and silhouette score by sample size."""
    sample_summary = (
        summary.groupby("sample_size")[
            ["ari_mean", "nmi_mean", "silhouette_score_mean"]
        ]
        .mean()
        .reset_index()
    )
    long_df = sample_summary.melt(
        id_vars="sample_size",
        var_name="metric",
        value_name="mean_value",
    )
    metric_names = {
        "ari_mean": "ARI",
        "nmi_mean": "NMI",
        "silhouette_score_mean": "Silhouette",
    }
    long_df["metric"] = long_df["metric"].map(metric_names)

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=long_df,
        x="sample_size",
        y="mean_value",
        hue="metric",
        marker="o",
        linewidth=1.8,
    )
    plt.xlabel("Sample Size")
    plt.ylabel("Mean Metric Value")
    plt.title("Simulation Metrics by Sample Size")
    plt.ylim(0, 1.05)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "simulation_metrics_by_sample_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_cluster_profile_heatmap(profile: pd.DataFrame) -> None:
    """Plot standardized true-cluster profiles for the representative dataset."""
    heatmap_data = profile.set_index("true_cluster")[SIMULATION_VARIABLES]
    standardized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0).replace(
        0, np.nan
    )
    standardized = standardized.fillna(0)
    standardized.index = [f"True Cluster {index}" for index in standardized.index]

    plt.figure(figsize=(13, 5))
    sns.heatmap(standardized, cmap="coolwarm", center=0, linewidths=0.3)
    plt.xlabel("Variable")
    plt.ylabel("True Cluster")
    plt.title("Simulation True Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "simulation_cluster_profile_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_simulation_results(
    summary: pd.DataFrame,
    representative_df: pd.DataFrame,
    representative_scores: pd.DataFrame,
    matched_labels: np.ndarray,
) -> None:
    """Create all required simulation figures."""
    plot_pca_cluster_scatter(
        scores=representative_scores,
        label_column="true_cluster",
        output_name="simulation_true_clusters_pca.png",
        title="Simulation PCA Scatter by True Clusters",
        legend_title="True Cluster",
    )
    plot_pca_cluster_scatter(
        scores=representative_scores,
        label_column="predicted_cluster",
        output_name="simulation_kmeans_clusters_pca.png",
        title="Simulation PCA Scatter by K-means Clusters",
        legend_title="K-means Cluster",
    )
    plot_simulation_confusion_matrix(
        true_labels=representative_df["true_cluster"].to_numpy(),
        matched_labels=matched_labels,
    )
    plot_metric_by_noise(
        summary=summary,
        metric_column="ari_mean",
        output_name="simulation_ari_by_noise.png",
        ylabel="Mean ARI",
        title="ARI by Noise Level",
    )
    plot_metric_by_noise(
        summary=summary,
        metric_column="nmi_mean",
        output_name="simulation_nmi_by_noise.png",
        ylabel="Mean NMI",
        title="NMI by Noise Level",
    )
    plot_metric_by_noise(
        summary=summary,
        metric_column="silhouette_score_mean",
        output_name="simulation_silhouette_by_noise.png",
        ylabel="Mean Silhouette Score",
        title="Silhouette Score by Noise Level",
    )
    plot_metrics_by_sample_size(summary)
    profile = pd.read_csv(TABLES_DIR / "simulation_cluster_profile.csv")
    plot_cluster_profile_heatmap(profile)


def write_simulation_summary(
    design: pd.DataFrame,
    summary: pd.DataFrame,
    representative_scores: pd.DataFrame,
) -> None:
    """Write a formal Chinese report-ready simulation summary."""
    low_noise = (
        summary.loc[summary["noise_level"] == min(NOISE_LEVELS)]
        [["ari_mean", "nmi_mean", "silhouette_score_mean"]]
        .mean()
    )
    high_noise = (
        summary.loc[summary["noise_level"] == max(NOISE_LEVELS)]
        [["ari_mean", "nmi_mean", "silhouette_score_mean"]]
        .mean()
    )
    by_sample = (
        summary.groupby("sample_size")[["ari_mean", "nmi_mean", "silhouette_score_mean"]]
        .mean()
        .reset_index()
    )
    best_sample = by_sample.sort_values("ari_mean", ascending=False).iloc[0]
    worst_sample = by_sample.sort_values("ari_mean", ascending=True).iloc[0]
    representative_components = int(representative_scores["n_components"].iloc[0])

    design_text = "\n".join(
        f"- 样本量 {int(row.sample_size)}，噪声水平 {row.noise_level:.1f}，"
        f"重复 {int(row.repetitions)} 次，真实类别数 {int(row.n_clusters)}"
        for row in design.itertuples(index=False)
    )

    sample_text = "\n".join(
        f"- n={int(row.sample_size)}：平均 ARI={row.ari_mean:.4f}，"
        f"平均 NMI={row.nmi_mean:.4f}，平均 silhouette={row.silhouette_score_mean:.4f}"
        for row in by_sample.itertuples(index=False)
    )

    summary_text = f"""# Simulation Study 仿真实验总结

## 仿真实验目的

本节仿真实验用于检验：当数据中真实存在不同信用卡客户群体时，本文采用的“标准化 + PCA 降维 + K-means 聚类”流程是否能够较好识别这些潜在群体。仿真实验不替代 Kaggle 实证分析，而是作为 Section 3: Simulation studies，为实证方法提供一个已知真实类别条件下的补充验证。

## 模拟数据设计思路

模拟数据包含与信用卡消费行为相似的 16 个数值变量，包括余额、购买金额、一次性消费、分期消费、取现、购买频率、取现频率、交易次数、信用额度、还款金额、最低还款额、全额还款比例和账户期限等。金额类和交易次数类变量在对数尺度上加入随机扰动，再转换回原始尺度，因此能够呈现类似金融数据的右偏分布；频率类变量通过 logit 和 logistic 转换限制在 0 到 1 之间；`TENURE` 被限制在 6 到 12 之间。

为使数据更接近真实金融行为，仿真并不是简单生成相互独立变量，而是通过共同因子结构引入变量相关性。例如购买金额、购买频率和购买交易次数之间正相关；取现金额、取现频率和取现次数之间正相关；信用额度、余额和还款金额之间也存在一定相关性。

## 四类模拟客户设定

本研究设定四类真实客户：

- 低活跃低消费客户：购买金额、交易次数和信用额度较低，取现行为也较弱。
- 高消费高信用额度客户：购买金额、一次性消费、信用额度和还款金额较高。
- 取现依赖型客户：取现金额、取现频率和取现次数较高，全额还款比例较低。
- 高频分期消费客户：购买频率和分期消费频率较高，分期消费金额占比较高。

这些类别具有明确业务解释，同时通过噪声项保留组内差异，使模拟数据不会呈现完全分离的理想状态。

## 实验参数设置

实验设置如下：

{design_text}

每次重复实验均先生成模拟数据，再进行标准化处理。PCA 保留累计解释方差达到约 80% 的主成分；代表性数据集中最终保留 {representative_components} 个主成分。随后在 PCA 得分上使用 K-means 聚类，聚类数固定为真实类别数 4。

## 评价指标

本文使用 ARI 和 NMI 评价聚类标签与真实标签的一致性。二者不受 K-means 标签编号任意性的影响，适合用于仿真实验。与此同时，还计算 silhouette score、Calinski-Harabasz index 和 Davies-Bouldin index，用于从簇内紧密度和簇间分离度角度评价聚类结构。对于混淆矩阵展示，本文使用 Hungarian algorithm 对 K-means 标签和真实标签进行最佳匹配。

## 主要结果解释

在低噪声水平（noise={min(NOISE_LEVELS):.1f}）下，平均 ARI 为 {low_noise["ari_mean"]:.4f}，平均 NMI 为 {low_noise["nmi_mean"]:.4f}，平均 silhouette score 为 {low_noise["silhouette_score_mean"]:.4f}。这说明当客户群体结构较清晰时，PCA + K-means 能够较好识别真实类别。

在高噪声水平（noise={max(NOISE_LEVELS):.1f}）下，平均 ARI 下降为 {high_noise["ari_mean"]:.4f}，平均 NMI 下降为 {high_noise["nmi_mean"]:.4f}，平均 silhouette score 为 {high_noise["silhouette_score_mean"]:.4f}。这表明当组内波动增强、客户群体之间出现更多重叠时，聚类识别能力会下降。

从样本量角度看，各样本量下的平均表现如下：

{sample_text}

其中 n={int(best_sample.sample_size)} 的平均 ARI 最高，n={int(worst_sample.sample_size)} 的平均 ARI 最低。总体来看，样本量增加有助于提高结果稳定性，但聚类效果的主要限制仍来自噪声水平和群体重叠程度。

## 结论

仿真实验表明，在真实存在较明显客户群体结构的情况下，标准化、PCA 降维与 K-means 聚类的组合能够较好恢复潜在类别；但当噪声较强或群体边界模糊时，ARI、NMI 和 silhouette score 均会下降。这一结果说明，在实证分析中不能只依赖单一聚类输出，而应结合内部评价指标、组间差异检验、判别分析结果和业务画像共同解释客户细分。

## 局限性

模拟数据的变量结构和客户类别由研究者人为设定，无法完全代表真实信用卡客户行为。真实数据中可能存在更复杂的非线性关系、异常消费模式、时间动态变化和更模糊的客户边界。因此，仿真实验只能说明方法在受控条件下的有效性，不能替代对真实数据结果的统计检验和业务解释。
"""
    (SUMMARIES_DIR / "simulation_study_summary.md").write_text(
        summary_text, encoding="utf-8"
    )


def main() -> None:
    """Run the complete simulation study workflow."""
    ensure_directories()
    design, _, summary = run_simulation_grid()
    representative_df, representative_scores, _, matched_labels = create_representative_outputs()
    plot_simulation_results(
        summary=summary,
        representative_df=representative_df,
        representative_scores=representative_scores,
        matched_labels=matched_labels,
    )
    write_simulation_summary(
        design=design,
        summary=summary,
        representative_scores=representative_scores,
    )
    print("Simulation study completed successfully.")


if __name__ == "__main__":
    main()
