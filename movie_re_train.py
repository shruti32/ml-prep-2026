# train.py
import numpy as np
import pandas as pd
import pickle
from src.svd_cf import fit

model = fit("ml-25m/ratings.csv", k=50, min_ratings_per_user=50)

with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to svd_model.pkl")
