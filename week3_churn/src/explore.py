"""
explore.py — Week 3 Telco Churn: Data Exploration
---------------------------------------------------
Run this before building the pipeline to understand the data.

Usage:
    python week3_churn/src/explore.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_PATH = pathlib.Path(__file__).parents[1] / "data" / "telco_churn.csv"
PLOT_DIR = pathlib.Path(__file__).parents[1] / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # TotalCharges is read as object because new customers have " " (a space).
    # Fix it here so the rest of the exploration uses numeric values.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    return df


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── 1. Basic shape & dtypes ───────────────────────────────────────────────────

def overview(df: pd.DataFrame) -> None:
    print_section("1. Shape & dtypes")
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print("\nDtype breakdown:")
    print(df.dtypes.value_counts().to_string())

    print("\nColumn list:")
    for col in df.columns:
        print(f"  {col:<25}  {df[col].dtype}   nunique={df[col].nunique()}")


# ── 2. Missing values ─────────────────────────────────────────────────────────

def missing_values(df: pd.DataFrame) -> None:
    print_section("2. Missing values")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("No missing values found.")
    else:
        print(nulls.to_string())
        print(f"\nTIP: TotalCharges nulls = new customers (tenure=0). "
              f"Fill with 0 before scaling.")


# ── 3. Target distribution ────────────────────────────────────────────────────

def target_distribution(df: pd.DataFrame) -> None:
    print_section("3. Target: Churn")
    counts = df["Churn"].value_counts()
    pcts   = df["Churn"].value_counts(normalize=True) * 100
    print(pd.concat([counts, pcts.round(1)], axis=1, keys=["count", "%"]).to_string())

    print(f"\nClass imbalance ratio (No:Yes): "
          f"{counts['No'] / counts['Yes']:.1f}:1")
    print("→ Use scale_pos_weight (XGBoost) or class_weight='balanced' (LGBM/RF)")

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    counts.plot(kind="bar", color=["steelblue", "tomato"], edgecolor="white", ax=ax)
    ax.set_title("Churn distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "target_distribution.png", dpi=120)
    print(f"\nPlot saved → {PLOT_DIR / 'target_distribution.png'}")


# ── 4. Numeric feature summary ────────────────────────────────────────────────

def numeric_summary(df: pd.DataFrame) -> None:
    print_section("4. Numeric features")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    print(df[num_cols].describe().round(2).to_string())

    print("\nObservations:")
    print("  • tenure range 0–72 months → good candidate for bucketing")
    print("  • MonthlyCharges range ~18–118 → scale before tree models? (not needed)")
    print("  • TotalCharges highly correlated with tenure × MonthlyCharges")

    # Pairplot of numerics vs Churn
    df_plot = df[num_cols + ["Churn"]].copy()
    df_plot["Churn_binary"] = (df_plot["Churn"] == "Yes").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, num_cols):
        sns.boxplot(data=df_plot, x="Churn", y=col, palette=["steelblue", "tomato"], ax=ax)
        ax.set_title(col)
    plt.suptitle("Numeric features vs Churn", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "numeric_vs_churn.png", dpi=120)
    print(f"\nPlot saved → {PLOT_DIR / 'numeric_vs_churn.png'}")


# ── 5. Tenure bucketing rationale ────────────────────────────────────────────

def tenure_analysis(df: pd.DataFrame) -> None:
    print_section("5. Tenure → churn rate by bucket")
    bins   = [0, 12, 24, 48, 72]
    labels = ["new (0–12)", "growing (13–24)", "established (25–48)", "loyal (49–72)"]
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

    summary = (
        df.groupby("tenure_group", observed=True)["Churn"]
        .value_counts(normalize=True)
        .unstack()
        .round(3)
    )
    print(summary.to_string())
    print("\nTIP: New customers churn at ~50% — tenure_group is a strong feature.")


# ── 6. Categorical features ───────────────────────────────────────────────────

def categorical_summary(df: pd.DataFrame) -> None:
    print_section("6. Categorical features — churn rates")
    cat_cols = df.select_dtypes(include="object").columns.drop(["customerID", "Churn"]).tolist()

    rows = []
    for col in cat_cols:
        for val, grp in df.groupby(col):
            churn_rate = (grp["Churn"] == "Yes").mean()
            rows.append({"feature": col, "value": val, "churn_rate": round(churn_rate, 3),
                         "n": len(grp)})
    summary = pd.DataFrame(rows).sort_values(["feature", "churn_rate"], ascending=[True, False])
    print(summary.to_string(index=False))

    print("\nKey observations:")
    print("  • Month-to-month contracts churn ~43% vs 3% for two-year")
    print("  • Fiber optic customers churn more (higher bill, more alternatives)")
    print("  • No OnlineSecurity/TechSupport → higher churn")


# ── 7. Charges ratio preview ─────────────────────────────────────────────────

def charges_ratio_preview(df: pd.DataFrame) -> None:
    print_section("7. Engineered feature preview: charges_ratio")
    df = df.copy()
    df["charges_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)

    print("charges_ratio = MonthlyCharges / TotalCharges")
    print("  High ratio → customer is relatively new (TotalCharges still low)")
    print("  Low ratio  → long-tenure customer; spend has accumulated\n")
    print(df.groupby("Churn")["charges_ratio"].describe().round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(data=df, x="charges_ratio", hue="Churn",
                palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
    ax.set_title("charges_ratio distribution by Churn")
    ax.set_xlim(0, None)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "charges_ratio_vs_churn.png", dpi=120)
    print(f"\nPlot saved → {PLOT_DIR / 'charges_ratio_vs_churn.png'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading data from: {DATA_PATH}")
    df = load_raw()

    overview(df)
    missing_values(df)
    target_distribution(df)
    numeric_summary(df)
    tenure_analysis(df)
    categorical_summary(df)
    charges_ratio_preview(df)

    print("\n\nExploration complete. Plots saved to:", PLOT_DIR)
    print("Key pipeline implications:")
    print("  1. Fix TotalCharges (coerce ' ' → 0) before any numeric step")
    print("  2. Add tenure_group as a categorical feature (TenureGrouper transformer)")
    print("  3. Add charges_ratio as a numeric feature (ChargesRatioAdder transformer)")
    print("  4. OHE all object columns (except customerID — drop it)")
    print("  5. Handle class imbalance via model parameter, not resampling")


if __name__ == "__main__":
    main()
