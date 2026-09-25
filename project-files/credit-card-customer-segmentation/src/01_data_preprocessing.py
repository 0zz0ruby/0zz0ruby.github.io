"""Data preprocessing for the MA304 credit card customer segmentation project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


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

AMOUNT_VARIABLES = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]


def ensure_directories() -> None:
    """Create output folders used by this script."""
    for directory in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, SUMMARIES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """Load the raw Kaggle CSV file."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw data file not found. Please place CC GENERAL.csv in data/."
        )
    return pd.read_csv(RAW_DATA_PATH)


def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value summary before imputation."""
    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_ratio": df.isna().mean().values,
        }
    )
    missing_summary.to_csv(TABLES_DIR / "missing_values_summary.csv", index=False)
    return missing_summary


def prepare_modeling_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove the customer ID and keep numeric modeling variables."""
    modeling_df = df.drop(columns=["CUST_ID"], errors="ignore").copy()
    numeric_columns = modeling_df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = [column for column in modeling_df.columns if column not in numeric_columns]

    if not numeric_columns:
        raise ValueError("No numeric modeling variables were found after removing CUST_ID.")

    return modeling_df[numeric_columns], excluded_columns


def impute_missing_values(modeling_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Impute numeric missing values using medians."""
    cleaned_df = modeling_df.copy()
    imputation_values: dict[str, float] = {}

    for column in cleaned_df.columns:
        if cleaned_df[column].isna().any():
            median_value = cleaned_df[column].median()
            if pd.isna(median_value):
                median_value = 0.0
            cleaned_df[column] = cleaned_df[column].fillna(median_value)
            imputation_values[column] = float(median_value)

    if cleaned_df.isna().any().any():
        remaining_columns = cleaned_df.columns[cleaned_df.isna().any()].tolist()
        raise ValueError(f"Missing values remain after imputation: {remaining_columns}")

    return cleaned_df, imputation_values


def save_descriptive_tables(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Save descriptive statistics and the correlation matrix."""
    descriptive_statistics = cleaned_df.describe().T
    descriptive_statistics["skewness"] = cleaned_df.skew(numeric_only=True)
    descriptive_statistics["kurtosis"] = cleaned_df.kurtosis(numeric_only=True)
    descriptive_statistics.to_csv(TABLES_DIR / "descriptive_statistics.csv")

    correlation_matrix = cleaned_df.corr()
    correlation_matrix.to_csv(TABLES_DIR / "correlation_matrix.csv")
    return correlation_matrix


def standardize_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize all modeling variables with StandardScaler."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(cleaned_df)
    return pd.DataFrame(scaled_array, columns=cleaned_df.columns, index=cleaned_df.index)


def available_key_variables(df: pd.DataFrame) -> list[str]:
    """Return key variables available in the current dataset."""
    selected = [column for column in KEY_VARIABLES if column in df.columns]
    if selected:
        return selected
    return df.columns[: min(12, len(df.columns))].tolist()


def plot_missing_values(missing_summary: pd.DataFrame) -> None:
    """Plot missing-value counts for all variables."""
    plt.figure(figsize=(10, 6))
    plot_data = missing_summary.sort_values("missing_count", ascending=False)

    if plot_data["missing_count"].sum() == 0:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        sns.barplot(data=plot_data, x="missing_count", y="column", color="#4C78A8")
        plt.xlabel("Missing Count")
        plt.ylabel("Variable")
        plt.title("Missing Values by Variable")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    """Plot a correlation heatmap for numeric variables."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def _series_for_visualization(df: pd.DataFrame, column: str) -> tuple[pd.Series, str]:
    """Use log1p for non-negative monetary variables to improve readability."""
    series = df[column]
    if column in AMOUNT_VARIABLES and series.min() >= 0:
        return np.log1p(series), f"log1p({column})"
    return series, column


def plot_distributions(cleaned_df: pd.DataFrame) -> None:
    """Plot distributions for key variables."""
    variables = available_key_variables(cleaned_df)
    n_cols = 3
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for axis, column in zip(axes, variables):
        series, label = _series_for_visualization(cleaned_df, column)
        sns.histplot(series, bins=30, kde=True, ax=axis, color="#4C78A8")
        axis.set_title(label)
        axis.set_xlabel(label)
        axis.set_ylabel("Count")

    for axis in axes[len(variables) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distributions_key_variables.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplots(cleaned_df: pd.DataFrame) -> None:
    """Plot boxplots for key variables without deleting extreme values."""
    variables = available_key_variables(cleaned_df)
    n_cols = 3
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for axis, column in zip(axes, variables):
        series, label = _series_for_visualization(cleaned_df, column)
        sns.boxplot(x=series, ax=axis, color="#F58518")
        axis.set_title(label)
        axis.set_xlabel(label)

    for axis in axes[len(variables) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_key_variables.png", dpi=300, bbox_inches="tight")
    plt.close()


def strongest_correlations(correlation_matrix: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the strongest absolute pairwise correlations."""
    corr = correlation_matrix.abs().where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    pairs = corr.stack().sort_values(ascending=False).head(top_n)
    return pairs.reset_index(name="absolute_correlation").rename(
        columns={"level_0": "variable_1", "level_1": "variable_2"}
    )


def write_data_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    imputation_values: dict[str, float],
    excluded_columns: list[str],
    correlation_matrix: pd.DataFrame,
) -> None:
    """Write a formal Chinese data description for the report."""
    missing_columns = missing_summary.loc[
        missing_summary["missing_count"] > 0, ["column", "missing_count", "missing_ratio"]
    ]
    if missing_columns.empty:
        missing_text = "原始数据中未发现缺失值。"
    else:
        missing_lines = [
            f"- `{row.column}`：缺失 {int(row.missing_count)} 个，占比 {row.missing_ratio:.2%}"
            for row in missing_columns.itertuples(index=False)
        ]
        missing_text = "\n".join(missing_lines)

    if imputation_values:
        imputation_lines = [
            f"- `{column}` 使用中位数 {value:.4f} 填补"
            for column, value in imputation_values.items()
        ]
        imputation_text = "\n".join(imputation_lines)
    else:
        imputation_text = "- 未发现需要填补的建模变量。"

    dtype_text = "\n".join(
        f"- {dtype}: {count} 个变量" for dtype, count in raw_df.dtypes.value_counts().items()
    )
    excluded_text = (
        "、".join(f"`{column}`" for column in excluded_columns)
        if excluded_columns
        else "无额外非数值变量被排除"
    )

    top_corr = strongest_correlations(correlation_matrix)
    top_corr_text = "\n".join(
        f"- `{row.variable_1}` 与 `{row.variable_2}`：|r| = {row.absolute_correlation:.3f}"
        for row in top_corr.itertuples(index=False)
    )

    summary = f"""# 数据描述与预处理总结

## 数据来源

本项目使用 Kaggle 数据集 **Credit Card Dataset for Clustering**，数据集标识为 `arjunbhasin2013/ccdata`。原始数据文件应命名为 `CC GENERAL.csv`，并放置在 `data/` 目录下。

## 样本和变量概况

原始数据共包含 {raw_df.shape[0]} 个客户样本、{raw_df.shape[1]} 个变量。变量类型概况如下：

{dtype_text}

`CUST_ID` 仅表示客户编号，不具有统计建模含义，因此已从 PCA、因子分析、聚类和判别分析的建模变量中删除。清洗后的建模数据包含 {cleaned_df.shape[1]} 个数值变量。{excluded_text}。

## 缺失值处理方式

缺失值检查结果如下：

{missing_text}

对于金融类变量，均值容易受到极端消费金额影响，因此本文使用中位数对缺失的数值变量进行填补。实际填补方式如下：

{imputation_text}

## 异常值处理思路

信用卡消费数据中的极端值可能代表高价值客户、大额消费客户或特殊取现行为，具有明确业务意义。因此本文没有简单删除异常值，而是通过箱线图识别变量分布特征，并在金额类变量的分布图中使用 `log1p` 辅助展示偏态分布。

## 标准化原因

不同变量的量纲差异较大，例如消费金额、信用额度、交易频率和还款比例不在同一尺度上。为了避免金额较大的变量在 PCA、K-means 聚类和 LDA 验证中占据不合理权重，本文使用 `StandardScaler` 对所有建模变量进行了标准化处理。清洗后的原始尺度数据保存为 `data/processed/credit_card_cleaned.csv`，标准化数据保存为 `data/processed/credit_card_scaled.csv`。

## 初步数据特征

从描述统计和可视化结果可以看出，金额类变量通常呈右偏分布，说明大多数客户消费金额较低，但少部分客户具有较高余额、较高购买额或较高取现金额。主要变量之间的较强相关关系包括：

{top_corr_text}

这些相关关系说明客户消费行为变量之间存在一定多重相关性，后续使用 PCA 和因子分析可以更好地提取综合行为维度。
"""
    (SUMMARIES_DIR / "data_description.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the full preprocessing workflow."""
    ensure_directories()
    raw_df = load_raw_data()

    missing_summary = inspect_missing_values(raw_df)
    modeling_df, excluded_columns = prepare_modeling_data(raw_df)
    cleaned_df, imputation_values = impute_missing_values(modeling_df)

    cleaned_df.to_csv(PROCESSED_DIR / "credit_card_cleaned.csv", index=False)
    correlation_matrix = save_descriptive_tables(cleaned_df)

    scaled_df = standardize_features(cleaned_df)
    scaled_df.to_csv(PROCESSED_DIR / "credit_card_scaled.csv", index=False)

    plot_missing_values(missing_summary)
    plot_correlation_heatmap(correlation_matrix)
    plot_distributions(cleaned_df)
    plot_boxplots(cleaned_df)

    write_data_summary(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        missing_summary=missing_summary,
        imputation_values=imputation_values,
        excluded_columns=excluded_columns,
        correlation_matrix=correlation_matrix,
    )

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
