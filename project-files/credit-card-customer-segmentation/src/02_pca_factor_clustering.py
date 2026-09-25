"""PCA, factor analysis, and K-means clustering for the MA304 project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "CC GENERAL.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"

RANDOM_STATE = 42

KEY_VARIABLES = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "TENURE",
]


def ensure_directories() -> None:
    """Create output folders used by this script."""
    for directory in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, SUMMARIES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, message: str) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(message)


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned and standardized data produced by preprocessing."""
    cleaned_path = PROCESSED_DIR / "credit_card_cleaned.csv"
    scaled_path = PROCESSED_DIR / "credit_card_scaled.csv"
    require_file(
        cleaned_path,
        "Processed data not found. Please run: python scripts/01_data_preprocessing.py",
    )
    require_file(
        scaled_path,
        "Scaled data not found. Please run: python scripts/01_data_preprocessing.py",
    )
    cleaned_df = pd.read_csv(cleaned_path)
    scaled_df = pd.read_csv(scaled_path)
    return cleaned_df, scaled_df


def select_pca_components(cumulative_variance: np.ndarray) -> int:
    """Select a PCA component count with cumulative variance around 70%-85%."""
    in_range = np.where((cumulative_variance >= 0.70) & (cumulative_variance <= 0.85))[0]
    if len(in_range) > 0:
        return int(in_range[0] + 1)

    at_least_80 = np.where(cumulative_variance >= 0.80)[0]
    if len(at_least_80) > 0:
        return int(at_least_80[0] + 1)

    return int(len(cumulative_variance))


def run_pca(scaled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Fit PCA, save tables and PCA scores."""
    pca = PCA()
    pca_scores_all = pca.fit_transform(scaled_df)

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
    explained_variance.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

    n_components = select_pca_components(
        explained_variance["cumulative_explained_variance_ratio"].to_numpy()
    )

    loading_values = pca.components_[:n_components].T * np.sqrt(
        pca.explained_variance_[:n_components]
    )
    loadings = pd.DataFrame(
        loading_values,
        index=scaled_df.columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings.to_csv(TABLES_DIR / "pca_loadings.csv")

    pca_scores = pd.DataFrame(
        pca_scores_all[:, :n_components],
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    pca_scores.to_csv(PROCESSED_DIR / "pca_scores.csv", index=False)

    return explained_variance, loadings, pca_scores, n_components


def plot_pca_results(
    explained_variance: pd.DataFrame,
    loadings: pd.DataFrame,
    pca_scores: pd.DataFrame,
    n_components: int,
) -> None:
    """Create PCA scree, cumulative variance, loading, and 2D score plots."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(explained_variance) + 1),
        explained_variance["eigenvalue"],
        marker="o",
        color="#4C78A8",
    )
    plt.axhline(1, color="#E45756", linestyle="--", label="Eigenvalue = 1")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.title("PCA Scree Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(explained_variance) + 1),
        explained_variance["cumulative_explained_variance_ratio"],
        marker="o",
        color="#54A24B",
    )
    plt.axhline(0.70, color="#F58518", linestyle="--", label="70%")
    plt.axhline(0.85, color="#E45756", linestyle="--", label="85%")
    plt.axvline(n_components, color="#4C78A8", linestyle=":", label=f"Selected: {n_components}")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Cumulative Explained Variance")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_cumulative_variance.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, max(6, 0.35 * len(loadings))))
    sns.heatmap(loadings, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("PCA Loading Heatmap")
    plt.xlabel("Principal Component")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_loading_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    if "PC2" in pca_scores.columns:
        plt.scatter(pca_scores["PC1"], pca_scores["PC2"], s=12, alpha=0.45, color="#4C78A8")
        plt.ylabel("PC2")
    else:
        plt.scatter(pca_scores["PC1"], np.zeros(len(pca_scores)), s=12, alpha=0.45)
        plt.ylabel("Zero Baseline")
    plt.xlabel("PC1")
    plt.title("PCA 2D Scatter")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_2d_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def component_interpretations(loadings: pd.DataFrame, max_components: int = 4) -> list[str]:
    """Summarize PCA components using variables with the largest absolute loadings."""
    interpretations = []
    for component in loadings.columns[:max_components]:
        top_variables = (
            loadings[component]
            .abs()
            .sort_values(ascending=False)
            .head(4)
            .index.tolist()
        )
        joined = "、".join(f"`{variable}`" for variable in top_variables)
        interpretations.append(f"- {component}：主要由 {joined} 等变量贡献较大。")
    return interpretations


def write_pca_summary(
    explained_variance: pd.DataFrame,
    loadings: pd.DataFrame,
    n_components: int,
) -> None:
    """Write a Chinese PCA summary."""
    cumulative = explained_variance.loc[
        n_components - 1, "cumulative_explained_variance_ratio"
    ]
    interpretation_text = "\n".join(component_interpretations(loadings))

    summary = f"""# PCA 主成分分析总结

## 方法目的

信用卡消费数据包含多项金额、频率和还款行为变量，变量之间可能存在明显相关性。主成分分析用于将原始标准化变量转换为少数彼此正交的综合指标，从而降低维度、缓解多重共线性，并为后续聚类提供更稳定的输入。

## 主成分选择

本研究先对全部标准化变量计算主成分，再根据累计解释方差选择主成分数量。最终选择前 {n_components} 个主成分，其累计解释方差为 {cumulative:.2%}，处于课程项目中常用的 70%–85% 解释区间内，能够在保留主要信息和降低维度之间取得平衡。

## 主成分含义

根据主成分载荷矩阵，前几个主成分的主要变量贡献如下：

{interpretation_text}

载荷较高的变量可以帮助解释每个主成分代表的消费行为维度，例如消费金额、取现依赖、信用额度和还款行为等。由于主成分是原始变量的线性组合，其具体含义需要结合载荷方向和变量业务含义共同判断。

## 对聚类分析的作用

后续 K-means 聚类使用 PCA 得分作为输入，而不是直接使用全部原始变量。这样可以减少变量间相关性和噪声对距离计算的影响，使客户细分结果更加稳定，也便于在二维主成分图中展示客户群体分布。
"""
    (SUMMARIES_DIR / "pca_summary.md").write_text(summary, encoding="utf-8")


def load_factor_analyzer():
    """Import factor_analyzer lazily and provide a clear error if missing."""
    try:
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import (
            calculate_bartlett_sphericity,
            calculate_kmo,
        )
    except ImportError as exc:
        raise ImportError(
            "factor_analyzer is required for factor analysis. "
            "Please run: pip install factor_analyzer"
        ) from exc
    return FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity


def choose_factor_count(eigenvalues: np.ndarray, n_variables: int) -> int:
    """Choose a readable factor count using eigenvalue > 1 with a cap."""
    count = int(np.sum(eigenvalues > 1))
    count = min(count, 6, n_variables)
    if count < 2 and n_variables >= 2:
        count = 2
    return count


def run_factor_analysis(
    scaled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Run KMO, Bartlett, and varimax-rotated factor analysis."""
    FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity = load_factor_analyzer()

    _, kmo_model = calculate_kmo(scaled_df)
    bartlett_chi_square, bartlett_p_value = calculate_bartlett_sphericity(scaled_df)

    factor_checker = FactorAnalyzer(rotation=None)
    factor_checker.fit(scaled_df)
    eigenvalues, _ = factor_checker.get_eigenvalues()
    n_factors = choose_factor_count(eigenvalues, scaled_df.shape[1])

    factor_model = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
    factor_model.fit(scaled_df)

    factor_columns = [f"Factor{i + 1}" for i in range(n_factors)]
    factor_loadings = pd.DataFrame(
        factor_model.loadings_,
        index=scaled_df.columns,
        columns=factor_columns,
    )
    factor_loadings.to_csv(TABLES_DIR / "factor_loadings.csv")

    variance_values = factor_model.get_factor_variance()
    factor_variance = pd.DataFrame(
        {
            "factor": factor_columns,
            "ss_loadings": variance_values[0],
            "proportion_variance": variance_values[1],
            "cumulative_variance": variance_values[2],
        }
    )
    factor_variance.to_csv(TABLES_DIR / "factor_variance.csv", index=False)

    tests = pd.DataFrame(
        {
            "metric": [
                "KMO",
                "Bartlett chi-square",
                "Bartlett p-value",
                "Selected factor count",
            ],
            "value": [kmo_model, bartlett_chi_square, bartlett_p_value, n_factors],
        }
    )
    tests.to_csv(TABLES_DIR / "factor_analysis_tests.csv", index=False)

    factor_names = infer_factor_names(factor_loadings)
    return tests, factor_loadings, factor_variance, factor_names


def infer_factor_names(factor_loadings: pd.DataFrame) -> dict[int, str]:
    """Assign interpretable Chinese names based on high-loading variables."""
    names: dict[int, str] = {}
    used_names: set[str] = set()

    for index, factor in enumerate(factor_loadings.columns, start=1):
        top_variables = (
            factor_loadings[factor]
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        variable_set = set(top_variables)

        if any("CASH_ADVANCE" in variable for variable in variable_set):
            proposed = "取现依赖因子"
        elif {"PURCHASES", "ONEOFF_PURCHASES"} & variable_set:
            proposed = "大额消费因子"
        elif {"PURCHASES_FREQUENCY", "INSTALLMENTS_PURCHASES"} & variable_set:
            proposed = "消费活跃度与分期倾向因子"
        elif {"CREDIT_LIMIT", "PAYMENTS"} & variable_set:
            proposed = "信用额度与偿付能力因子"
        elif {"BALANCE", "MINIMUM_PAYMENTS"} & variable_set:
            proposed = "余额与还款压力因子"
        elif "PRC_FULL_PAYMENT" in variable_set:
            proposed = "全额还款行为因子"
        else:
            proposed = "综合行为因子"

        if proposed in used_names:
            proposed = f"{proposed}{index}"
        used_names.add(proposed)
        names[index] = proposed

    return names


def plot_factor_heatmap(factor_loadings: pd.DataFrame) -> None:
    """Plot factor loading heatmap."""
    plt.figure(figsize=(10, max(6, 0.35 * len(factor_loadings))))
    sns.heatmap(factor_loadings, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Factor Loading Heatmap")
    plt.xlabel("Factor")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "factor_loading_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_factor_summary(
    tests: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    factor_variance: pd.DataFrame,
    factor_names: dict[int, str],
) -> None:
    """Write a Chinese factor analysis summary."""
    test_values = tests.set_index("metric")["value"]
    kmo_value = float(test_values["KMO"])
    bartlett_chi = float(test_values["Bartlett chi-square"])
    bartlett_p = float(test_values["Bartlett p-value"])
    n_factors = int(float(test_values["Selected factor count"]))

    suitability = (
        "KMO 值较高，Bartlett 球形检验显著，说明变量之间存在适合提取公共因子的相关结构。"
        if kmo_value >= 0.6 and bartlett_p < 0.05
        else "KMO 或 Bartlett 结果提示因子分析适用性需要谨慎解释，但仍可作为探索性结构分析参考。"
    )

    factor_lines = []
    for index, factor in enumerate(factor_loadings.columns, start=1):
        top_variables = (
            factor_loadings[factor]
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        factor_lines.append(
            f"- {factor}（{factor_names[index]}）：高载荷变量包括 "
            + "、".join(f"`{variable}`" for variable in top_variables)
            + "。"
        )

    cumulative_variance = factor_variance["cumulative_variance"].iloc[-1]
    factor_text = "\n".join(factor_lines)

    summary = f"""# 因子分析总结

## 适用性检验

KMO 检验值为 {kmo_value:.4f}。Bartlett 球形检验的卡方统计量为 {bartlett_chi:.4f}，p 值为 {bartlett_p:.4g}。{suitability}

## 因子提取结果

本文以特征值大于 1 为主要标准，并结合解释性限制因子数量，最终提取 {n_factors} 个公共因子，累计解释方差为 {cumulative_variance:.2%}。为提高因子含义的清晰度，模型使用 varimax 正交旋转。

## 因子命名与含义

{factor_text}

这些公共因子从潜在行为维度解释了客户差异。与 PCA 相比，因子分析更强调可解释的潜变量结构，因此可以辅助说明聚类结果背后的消费行为逻辑。
"""
    (SUMMARIES_DIR / "factor_analysis_summary.md").write_text(summary, encoding="utf-8")


def evaluate_kmeans(pca_scores: pd.DataFrame) -> pd.DataFrame:
    """Evaluate K-means for k from 2 to 10."""
    max_k = min(10, len(pca_scores) - 1)
    if max_k < 2:
        raise ValueError("At least three observations are required for K-means validation.")

    records = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(pca_scores)
        records.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(pca_scores, labels),
                "calinski_harabasz_index": calinski_harabasz_score(pca_scores, labels),
                "davies_bouldin_index": davies_bouldin_score(pca_scores, labels),
            }
        )

    metrics = pd.DataFrame(records)
    metrics["silhouette_rank"] = metrics["silhouette_score"].rank(
        ascending=False, method="min"
    )
    metrics["calinski_harabasz_rank"] = metrics["calinski_harabasz_index"].rank(
        ascending=False, method="min"
    )
    metrics["davies_bouldin_rank"] = metrics["davies_bouldin_index"].rank(
        ascending=True, method="min"
    )
    metrics["combined_rank_score"] = (
        metrics["silhouette_rank"]
        + metrics["calinski_harabasz_rank"]
        + metrics["davies_bouldin_rank"]
    )

    metrics.to_csv(TABLES_DIR / "clustering_model_selection.csv", index=False)
    metrics.to_csv(TABLES_DIR / "clustering_metrics.csv", index=False)
    return metrics


def select_best_k(metrics: pd.DataFrame) -> int:
    """Choose the final k using metric ranks with an interpretability preference for k=4."""
    ranked = metrics.sort_values(["combined_rank_score", "davies_bouldin_index", "k"])
    best_metric_k = int(ranked.iloc[0]["k"])
    max_silhouette = metrics["silhouette_score"].max()
    best_db = metrics["davies_bouldin_index"].min()

    k4 = metrics.loc[metrics["k"] == 4]
    if not k4.empty:
        k4_row = k4.iloc[0]
        if (
            k4_row["silhouette_score"] >= 0.90 * max_silhouette
            and k4_row["davies_bouldin_index"] <= 1.20 * best_db
        ):
            return 4

    return best_metric_k


def fit_final_kmeans(pca_scores: pd.DataFrame, best_k: int) -> tuple[np.ndarray, KMeans]:
    """Fit the final K-means model."""
    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(pca_scores)
    return labels, model


def available_profile_variables(cleaned_df: pd.DataFrame) -> list[str]:
    """Return key variables available for cluster profiling."""
    variables = [column for column in KEY_VARIABLES if column in cleaned_df.columns]
    if variables:
        return variables
    return cleaned_df.columns[: min(12, len(cleaned_df.columns))].tolist()


def infer_cluster_names(cluster_profile: pd.DataFrame, variables: list[str]) -> dict[int, str]:
    """Name clusters according to their original-scale profiles."""
    profile_indexed = cluster_profile.set_index("Cluster")
    means = profile_indexed[variables]
    z_means = (means - means.mean()) / means.std(ddof=0).replace(0, np.nan)
    z_means = z_means.fillna(0)

    candidates = {
        "低活跃低消费客户": lambda row: -np.mean(
            [
                row.get("BALANCE", 0),
                row.get("PURCHASES", 0),
                row.get("CASH_ADVANCE", 0),
                row.get("PAYMENTS", 0),
            ]
        ),
        "高消费高信用额度客户": lambda row: np.mean(
            [
                row.get("PURCHASES", 0),
                row.get("ONEOFF_PURCHASES", 0),
                row.get("CREDIT_LIMIT", 0),
                row.get("PAYMENTS", 0),
            ]
        ),
        "取现依赖型客户": lambda row: np.mean(
            [
                row.get("CASH_ADVANCE", 0),
                row.get("CASH_ADVANCE_FREQUENCY", 0),
            ]
        ),
        "高频分期消费客户": lambda row: np.mean(
            [
                row.get("INSTALLMENTS_PURCHASES", 0),
                row.get("PURCHASES_FREQUENCY", 0),
            ]
        ),
        "高余额高还款压力客户": lambda row: np.mean(
            [
                row.get("BALANCE", 0),
                row.get("MINIMUM_PAYMENTS", 0),
                -row.get("PRC_FULL_PAYMENT", 0),
            ]
        ),
        "稳定还款型客户": lambda row: np.mean(
            [
                row.get("PAYMENTS", 0),
                row.get("PRC_FULL_PAYMENT", 0),
                row.get("CREDIT_LIMIT", 0),
            ]
        ),
    }

    cluster_names: dict[int, str] = {}
    used_names: set[str] = set()
    for cluster, row in z_means.iterrows():
        scored_names = sorted(
            ((name, scorer(row)) for name, scorer in candidates.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        for name, _ in scored_names:
            if name not in used_names:
                cluster_names[int(cluster)] = name
                used_names.add(name)
                break
        else:
            cluster_names[int(cluster)] = f"综合行为客户群{int(cluster)}"

    return cluster_names


def profile_clusters(
    cleaned_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Create cluster size, original-scale profile, and scaled center tables."""
    variables = available_profile_variables(cleaned_df)
    labeled_cleaned = cleaned_df.copy()
    labeled_cleaned["Cluster"] = labels

    size_summary = (
        labeled_cleaned["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="count")
    )
    size_summary["percentage"] = size_summary["count"] / len(labeled_cleaned)

    profile = labeled_cleaned.groupby("Cluster")[variables].mean().reset_index()
    profile = size_summary.merge(profile, on="Cluster", how="left")
    cluster_names = infer_cluster_names(profile, variables)
    profile.insert(1, "Cluster_Name", profile["Cluster"].map(cluster_names))
    profile.to_csv(TABLES_DIR / "cluster_profile_original_scale.csv", index=False)

    size_summary.insert(1, "Cluster_Name", size_summary["Cluster"].map(cluster_names))
    size_summary.to_csv(TABLES_DIR / "cluster_size_summary.csv", index=False)

    labeled_scaled = scaled_df.copy()
    labeled_scaled["Cluster"] = labels
    scaled_centers = labeled_scaled.groupby("Cluster").mean().reset_index()
    scaled_centers.insert(1, "Cluster_Name", scaled_centers["Cluster"].map(cluster_names))
    scaled_centers.to_csv(TABLES_DIR / "cluster_centers_scaled.csv", index=False)

    clustered_customers = cleaned_df.copy()
    clustered_customers["Cluster"] = labels
    clustered_customers["Cluster_Name"] = clustered_customers["Cluster"].map(cluster_names)
    clustered_customers.to_csv(PROCESSED_DIR / "clustered_customers.csv", index=False)

    return profile, size_summary, scaled_centers, cluster_names


def plot_clustering_results(
    metrics: pd.DataFrame,
    pca_scores: pd.DataFrame,
    labels: np.ndarray,
    profile: pd.DataFrame,
    size_summary: pd.DataFrame,
    scaled_centers: pd.DataFrame,
    best_k: int,
) -> None:
    """Create all required clustering figures."""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["k"], metrics["inertia"], marker="o", color="#4C78A8")
    plt.axvline(best_k, color="#E45756", linestyle="--", label=f"Selected k = {best_k}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["k"], metrics["silhouette_score"], marker="o", color="#54A24B")
    plt.axvline(best_k, color="#E45756", linestyle="--", label=f"Selected k = {best_k}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "silhouette_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    scatter_df = pca_scores.copy()
    scatter_df["Cluster"] = labels.astype(str)
    plt.figure(figsize=(9, 7))
    y_values = scatter_df["PC2"] if "PC2" in scatter_df.columns else np.zeros(len(scatter_df))
    sns.scatterplot(
        x=scatter_df["PC1"],
        y=y_values,
        hue=scatter_df["Cluster"],
        palette="tab10",
        s=16,
        alpha=0.55,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2" if "PC2" in scatter_df.columns else "Zero Baseline")
    plt.title("Cluster Scatter Plot on PCA Scores")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    size_plot = size_summary.copy()
    size_plot["Cluster_Label"] = size_plot["Cluster"].map(lambda value: f"Cluster {value}")
    sns.barplot(
        data=size_plot,
        x="Cluster_Label",
        y="count",
        hue="Cluster_Label",
        palette="Set2",
        legend=False,
    )
    plt.xlabel("Cluster")
    plt.ylabel("Customer Count")
    plt.title("Cluster Size")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_size_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    heatmap_data = scaled_centers.copy()
    heatmap_data["Cluster_Label"] = heatmap_data["Cluster"].map(
        lambda value: f"Cluster {value}"
    )
    heatmap_data = heatmap_data.set_index("Cluster_Label").drop(
        columns=["Cluster", "Cluster_Name"]
    )
    plt.figure(figsize=(12, max(5, 0.7 * len(heatmap_data))))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, linewidths=0.3)
    plt.xlabel("Standardized Variable")
    plt.ylabel("Cluster")
    plt.title("Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_profile_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plot_cluster_radar(profile)


def plot_cluster_radar(profile: pd.DataFrame) -> None:
    """Plot a radar chart using min-max scaled original profile variables."""
    variables = [column for column in profile.columns if column not in ["Cluster", "Cluster_Name", "count", "percentage"]]
    radar_data = profile[variables].copy()
    ranges = radar_data.max() - radar_data.min()
    radar_scaled = (radar_data - radar_data.min()) / ranges.replace(0, np.nan)
    radar_scaled = radar_scaled.fillna(0.5)

    angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
    angles += angles[:1]

    fig, axis = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for idx, row in radar_scaled.iterrows():
        values = row.tolist()
        values += values[:1]
        label = f"Cluster {profile.loc[idx, 'Cluster']}"
        axis.plot(angles, values, linewidth=1.6, label=label)
        axis.fill(angles, values, alpha=0.08)

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(variables, fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_title("Cluster Profile Radar")
    axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_profile_radar.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_clustering_summary(
    metrics: pd.DataFrame,
    best_k: int,
    profile: pd.DataFrame,
) -> None:
    """Write a Chinese clustering summary."""
    best_row = metrics.loc[metrics["k"] == best_k].iloc[0]
    metric_lines = "\n".join(
        f"- k={int(row.k)}：silhouette={row.silhouette_score:.4f}，"
        f"Calinski-Harabasz={row.calinski_harabasz_index:.2f}，"
        f"Davies-Bouldin={row.davies_bouldin_index:.4f}"
        for row in metrics.itertuples(index=False)
    )

    cluster_lines = []
    for row in profile.itertuples(index=False):
        core_values = []
        for variable in ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT"]:
            if variable in profile.columns:
                core_values.append(f"{variable}={getattr(row, variable):.2f}")
        cluster_lines.append(
            f"- Cluster {int(row.Cluster)}（{row.Cluster_Name}）："
            f"{int(row.count)} 人，占比 {row.percentage:.2%}；"
            + "，".join(core_values)
            + "。"
        )

    profile_text = "\n".join(cluster_lines)

    summary = f"""# K-means 聚类分析总结

## 方法说明

K-means 聚类适用于基于连续变量距离的客户细分。本文没有直接在原始变量上聚类，而是使用 PCA 得分作为输入，以降低变量间多重相关性和噪声对欧氏距离的影响。

## 聚类数选择

本文比较了 k=2 到 k=10 的聚类结果，并同时参考 inertia、silhouette score、Calinski-Harabasz index 和 Davies-Bouldin index。各候选 k 的主要指标如下：

{metric_lines}

综合内部评价指标和客户画像可解释性，最终选择 k={best_k}。该方案的 silhouette score 为 {best_row.silhouette_score:.4f}，Calinski-Harabasz index 为 {best_row.calinski_harabasz_index:.2f}，Davies-Bouldin index 为 {best_row.davies_bouldin_index:.4f}。

## 客户群体特征

聚类画像基于清洗后的原始尺度变量计算，而不是仅使用标准化变量。各客户群体概况如下：

{profile_text}

## 业务意义

聚类结果显示，信用卡客户可以按照消费金额、取现行为、信用额度和还款习惯划分为具有差异的群体。对于低活跃客户，银行可通过小额优惠和活跃度提升活动促进使用；对于高消费或高信用额度客户，可重点维护并提供权益服务；对于取现依赖或高余额压力客户，应关注潜在信用风险和还款压力；对于分期倾向明显的客户，可设计更有针对性的分期产品。
"""
    (SUMMARIES_DIR / "clustering_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run PCA, factor analysis, and clustering."""
    ensure_directories()
    cleaned_df, scaled_df = load_processed_data()

    explained_variance, pca_loadings, pca_scores, n_components = run_pca(scaled_df)
    plot_pca_results(explained_variance, pca_loadings, pca_scores, n_components)
    write_pca_summary(explained_variance, pca_loadings, n_components)

    factor_tests, factor_loadings, factor_variance, factor_names = run_factor_analysis(scaled_df)
    plot_factor_heatmap(factor_loadings)
    write_factor_summary(factor_tests, factor_loadings, factor_variance, factor_names)

    clustering_metrics = evaluate_kmeans(pca_scores)
    best_k = select_best_k(clustering_metrics)
    labels, _ = fit_final_kmeans(pca_scores, best_k)
    profile, size_summary, scaled_centers, _ = profile_clusters(cleaned_df, scaled_df, labels)
    plot_clustering_results(
        metrics=clustering_metrics,
        pca_scores=pca_scores,
        labels=labels,
        profile=profile,
        size_summary=size_summary,
        scaled_centers=scaled_centers,
        best_k=best_k,
    )
    write_clustering_summary(clustering_metrics, best_k, profile)

    print("PCA, factor analysis, and clustering completed successfully.")


if __name__ == "__main__":
    main()
