import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_churn_balance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts = df["Churn"].value_counts(normalize=True)
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax, palette=["#4c72b0", "#dd8452"])
    ax.set_title("Churn rate distribution")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Churn")
    return fig


def plot_feature_importance(df: pd.DataFrame, top_n: int = 10):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="importance_normalized", y="feature", data=df.head(top_n), ax=ax, palette="viridis")
    ax.set_title("Top feature importance")
    ax.set_xlabel("Normalized importance")
    return fig


def plot_correlation_matrix(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=["number"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric feature correlation")
    return fig
