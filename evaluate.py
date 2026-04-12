"""
Evaluate SVD collaborative filter at different values of k.
Shows how latent dimension count affects prediction quality.
"""

import pandas as pd
from src.svd_cf import load_data, train_test_split, fit, evaluate_rmse

RATINGS_PATH = "ml-25m/ratings.csv"
K_VALUES = [10, 20, 50, 100]

print("Loading data...")
df = load_data(RATINGS_PATH, min_ratings_per_user=50)

print("Splitting into train / test (80/20 per user)...")
train_df, test_df = train_test_split(df, test_frac=0.2)

print(f"Train ratings: {len(train_df):,}  |  Test ratings: {len(test_df):,}\n")

results = []

for k in K_VALUES:
    print(f"── k={k} ──────────────────────────")
    model = fit(RATINGS_PATH, k=k, min_ratings_per_user=50, _df=train_df)
    rmse = evaluate_rmse(model, test_df)
    print(f"   RMSE: {rmse:.4f}\n")
    results.append({"k": k, "rmse": rmse})

print("═══════════════════════════════")
print(f"{'k':>6}  {'RMSE':>8}")
print("───────────────────────────────")
for r in results:
    print(f"{r['k']:>6}  {r['rmse']:>8.4f}")
