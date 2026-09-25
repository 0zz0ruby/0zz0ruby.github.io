"""Validation and final visualization checks for the MA304 project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split


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
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

REQUIRED_FIGURES = [
    "missing_values.png",
    "correlation_heatmap.png",
    "distributions_key_variables.png",
    "boxplots_key_variables.png",
    "pca_scree_plot.png",
    "pca_cumulative_variance.png",
    "pca_loading_heatmap.png",
    "pca_2d_scatter.png",
    "factor_loading_heatmap.png",
    "elbow_plot.png",
    "silhouette_plot.png",
    "cluster_pca_scatter.png",
    "cluster_size_bar.png",
    "cluster_profile_heatmap.png",
    "cluster_profile_radar.png",
    "confusion_matrix_heatmap.png",
]


def ensure_directories() -> None:
    """Create output folders used by this script."""
    for directory in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, SUMMARIES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, message: str) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(message)


def load_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned data, scaled data, PCA scores, and clustered customers."""
    cleaned_path = PROCESSED_DIR / "credit_card_cleaned.csv"
    scaled_path = PROCESSED_DIR / "credit_card_scaled.csv"
    pca_path = PROCESSED_DIR / "pca_scores.csv"
    clustered_path = PROCESSED_DIR / "clustered_customers.csv"

    require_file(
        cleaned_path,
        "Cleaned data not found. Please run: python scripts/01_data_preprocessing.py",
    )
    require_file(
        scaled_path,
        "Scaled data not found. Please run: python scripts/01_data_preprocessing.py",
    )
    require_file(
        pca_path,
        "PCA scores not found. Please run: python scripts/02_pca_factor_clustering.py",
    )
    require_file(
        clustered_path,
        "Clustered customers not found. Please run: python scripts/02_pca_factor_clustering.py",
    )

    return (
        pd.read_csv(cleaned_path),
        pd.read_csv(scaled_path),
        pd.read_csv(pca_path),
        pd.read_csv(clustered_path),
    )


def calculate_validation_metrics(pca_scores: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Calculate internal clustering validation metrics for the final labels."""
    feature_columns = [column for column in pca_scores.columns if column.startswith("PC")]
    x_pca = pca_scores[feature_columns]

    metrics = pd.DataFrame(
        {
            "metric": [
                "silhouette_score",
                "calinski_harabasz_index",
                "davies_bouldin_index",
            ],
            "value": [
                silhouette_score(x_pca, labels),
                calinski_harabasz_score(x_pca, labels),
                davies_bouldin_score(x_pca, labels),
            ],
        }
    )
    metrics.to_csv(TABLES_DIR / "validation_metrics.csv", index=False)
    return metrics


def run_group_difference_tests(
    cleaned_df: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Run Kruskal-Wallis tests for key variables across clusters."""
    available_variables = [variable for variable in KEY_VARIABLES if variable in cleaned_df.columns]
    records = []

    for variable in available_variables:
        groups = [
            cleaned_df.loc[labels == cluster, variable].dropna()
            for cluster in sorted(labels.unique())
        ]
        groups = [group for group in groups if len(group) > 0]
        if len(groups) < 2:
            statistic = np.nan
            p_value = np.nan
        else:
            statistic, p_value = kruskal(*groups)

        records.append(
            {
                "variable": variable,
                "statistic": statistic,
                "p_value": p_value,
                "significant_at_0_05": bool(pd.notna(p_value) and p_value < 0.05),
            }
        )

    results = pd.DataFrame(records)
    results.to_csv(TABLES_DIR / "group_difference_tests.csv", index=False)
    return results


def run_lda_validation(
    scaled_df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Use LDA to test whether clusters can be distinguished by original variables."""
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_df,
        labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    predictions = lda.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    min_cluster_size = int(labels.value_counts().min())
    if min_cluster_size >= 2:
        cv_folds = min(5, min_cluster_size)
        cv_scores = cross_val_score(lda, scaled_df, labels, cv=cv_folds)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
    else:
        cv_folds = np.nan
        cv_mean = np.nan
        cv_std = np.nan

    label_order = sorted(labels.unique())
    confusion = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=label_order),
        index=[f"Actual_{label}" for label in label_order],
        columns=[f"Predicted_{label}" for label in label_order],
    )
    confusion.to_csv(TABLES_DIR / "confusion_matrix.csv")

    report = classification_report(y_test, predictions, digits=4)
    report_text = (
        "Linear Discriminant Analysis validation\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Cross-validation folds: {cv_folds}\n"
        f"Cross-validation mean accuracy: {cv_mean:.4f}\n"
        f"Cross-validation std: {cv_std:.4f}\n\n"
        f"{report}"
    )
    (SUMMARIES_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (TABLES_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    validation = pd.DataFrame(
        {
            "metric": [
                "test_accuracy",
                "cross_validation_folds",
                "cross_validation_mean_accuracy",
                "cross_validation_std",
            ],
            "value": [accuracy, cv_folds, cv_mean, cv_std],
        }
    )
    validation.to_csv(TABLES_DIR / "lda_validation_metrics.csv", index=False)

    return validation, confusion, report_text


def plot_confusion_matrix(confusion: pd.DataFrame) -> None:
    """Plot the LDA confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", linewidths=0.3)
    plt.xlabel("Predicted Cluster")
    plt.ylabel("Actual Cluster")
    plt.title("LDA Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def check_required_figures() -> tuple[list[str], list[str]]:
    """Check which required figures exist."""
    existing = []
    missing = []
    for figure_name in REQUIRED_FIGURES:
        if (FIGURES_DIR / figure_name).exists():
            existing.append(figure_name)
        else:
            missing.append(figure_name)
    return existing, missing


def write_validation_summary(
    validation_metrics: pd.DataFrame,
    group_tests: pd.DataFrame,
    lda_metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    existing_figures: list[str],
    missing_figures: list[str],
) -> None:
    """Write a Chinese validation and visualization summary."""
    metric_values = validation_metrics.set_index("metric")["value"]
    lda_values = lda_metrics.set_index("metric")["value"]
    significant_count = int(group_tests["significant_at_0_05"].sum())
    total_tests = len(group_tests)

    significant_variables = group_tests.loc[
        group_tests["significant_at_0_05"], "variable"
    ].tolist()
    significant_text = (
        "、".join(f"`{variable}`" for variable in significant_variables)
        if significant_variables
        else "无变量在 0.05 水平显著"
    )

    confusion_total = confusion.to_numpy().sum()
    correct_total = np.trace(confusion.to_numpy())
    confusion_accuracy = correct_total / confusion_total if confusion_total else np.nan

    existing_text = "\n".join(f"- {figure}" for figure in existing_figures)
    missing_text = (
        "\n".join(f"- {figure}" for figure in missing_figures)
        if missing_figures
        else "- 所有要求图表均已生成。"
    )

    summary = f"""# 聚类验证与最终可视化总结

## 聚类结果可靠性

基于最终聚类标签和 PCA 得分，silhouette score 为 {metric_values["silhouette_score"]:.4f}，Calinski-Harabasz index 为 {metric_values["calinski_harabasz_index"]:.2f}，Davies-Bouldin index 为 {metric_values["davies_bouldin_index"]:.4f}。这些指标从簇内紧密度和簇间分离度角度评价聚类效果，可作为最终聚类数合理性的内部证据。

## 组间差异检验

考虑到信用卡金额类变量通常具有偏态分布，本文使用 Kruskal-Wallis 检验比较不同客户群体在关键变量上的差异。在 {total_tests} 个关键变量中，有 {significant_count} 个变量在 0.05 显著性水平下表现出组间差异，显著变量包括：{significant_text}。这说明聚类结果不仅是算法划分，也反映了客户行为变量上的统计差异。

## LDA 判别分析验证

本文进一步使用线性判别分析检验聚类标签能否由原始标准化变量较好地区分。测试集准确率为 {lda_values["test_accuracy"]:.4f}，交叉验证平均准确率为 {lda_values["cross_validation_mean_accuracy"]:.4f}。若准确率较高，说明不同客户群体在原始变量空间中具有较清晰的判别边界，聚类结果具有一定稳定性和可解释性。

## 混淆矩阵解释

混淆矩阵中对角线元素表示被 LDA 正确判别的客户数量，非对角线元素表示不同聚类之间的混淆情况。本次测试集中根据混淆矩阵计算得到的准确率为 {confusion_accuracy:.4f}。如果某两个客户群体之间存在较多混淆，通常说明它们在部分消费或还款行为上相似，需要在报告中结合聚类画像进一步解释。

## 可视化结果说明

已检查的图表文件如下：

{existing_text}

仍缺失的图表如下：

{missing_text}

这些图表覆盖了缺失值、相关矩阵、变量分布、PCA、因子载荷、聚类选择、客户画像和判别分析结果，能够支持成员 4 撰写最终报告和制作展示材料。

## 研究局限性

本研究使用的是横截面信用卡客户行为数据，聚类结果反映的是样本期内的相对行为差异，不能直接解释因果关系。K-means 聚类依赖欧氏距离和球形簇假设，对变量标准化、异常值和主成分数量选择较敏感。因子命名和客户群体命名也具有一定解释性判断，需要结合业务背景谨慎表述。
"""
    (SUMMARIES_DIR / "validation_visualization_summary.md").write_text(
        summary, encoding="utf-8"
    )


def main() -> None:
    """Run validation and final visualization checks."""
    ensure_directories()
    cleaned_df, scaled_df, pca_scores, clustered_customers = load_analysis_data()
    labels = clustered_customers["Cluster"]

    validation_metrics = calculate_validation_metrics(pca_scores, labels)
    group_tests = run_group_difference_tests(cleaned_df, labels)
    lda_metrics, confusion, _ = run_lda_validation(scaled_df, labels)
    plot_confusion_matrix(confusion)
    existing_figures, missing_figures = check_required_figures()
    write_validation_summary(
        validation_metrics=validation_metrics,
        group_tests=group_tests,
        lda_metrics=lda_metrics,
        confusion=confusion,
        existing_figures=existing_figures,
        missing_figures=missing_figures,
    )

    print("Validation and visualization checks completed successfully.")


if __name__ == "__main__":
    main()
