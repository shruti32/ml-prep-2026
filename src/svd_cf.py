import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from dataclasses import dataclass


@dataclass
class SVDModel:
    U: np.ndarray  # (n_users, k)  — user latent factors
    S: np.ndarray  # (k,)          — singular values
    Vt: np.ndarray  # (k, n_items)  — item latent factors
    user_index: dict  # original userId → row index
    item_index: dict  # original movieId → col index
    index_to_item: dict  # col index → movieId
    mean_rating: float
    user_means: np.ndarray


def load_data(ratings_path: str, min_ratings_per_user: int = 50) -> pd.DataFrame:
    """Load and filter ratings. Keep only active users for tractable matrix."""
    print("Loading ratings...")
    df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])

    # Keep users with enough ratings — reduces matrix size substantially
    user_counts = df["userId"].value_counts()
    active_users = user_counts[user_counts >= min_ratings_per_user].index
    df = df[df["userId"].isin(active_users)]

    print(
        f"Users: {df['userId'].nunique():,}  |  Movies: {df['movieId'].nunique():,}  |  Ratings: {len(df):,}"
    )
    return df


# def build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, dict, dict, dict, np.ndarray]:
#     """Build dense user-item matrix and index mappings."""
#     users = sorted(df["userId"].unique())
#     items = sorted(df["movieId"].unique())

#     user_index = {u: i for i, u in enumerate(users)}
#     item_index = {m: j for j, m in enumerate(items)}
#     index_to_item = {j: m for m, j in item_index.items()}

#     n_users, n_items = len(users), len(items)
#     print(f"Matrix shape: {n_users} × {n_items}")

#     R = np.zeros((n_users, n_items), dtype=np.float32)
#     for row in df.itertuples(index=False):
#         R[user_index[row.userId], item_index[row.movieId]] = row.rating

#     # Mean-centre per user (subtract each user's average rating)
#     user_means = np.true_divide(
#         R.sum(axis=1), np.maximum((R != 0).sum(axis=1), 1)
#     ).astype(np.float32)

#     # Subtract user means only from rated entries
#     mask = R != 0
#     R[mask] -= user_means[np.where(mask)[0]]

#     return R, user_index, item_index, index_to_item, user_means


def build_sparse_matrix(
    df: pd.DataFrame,
) -> tuple[sp.csr_matrix, dict, dict, dict, np.ndarray]:
    """
    Build a SPARSE user-item matrix.
    Only stores the ~25M observed ratings, not the billions of zeros.
    Memory: ~few hundred MB instead of 22 GB.
    """
    users = sorted(df["userId"].unique())
    items = sorted(df["movieId"].unique())

    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: j for j, m in enumerate(items)}
    index_to_item = {j: m for m, j in item_index.items()}

    n_users, n_items = len(users), len(items)
    print(
        f"Matrix shape: {n_users:,} × {n_items:,}  (dense would be {n_users * n_items * 4 / 1e9:.1f} GB)"
    )

    # Per-user mean from observed entries only
    user_mean_series = df.groupby("userId")["rating"].mean()
    user_means = np.array([user_mean_series[u] for u in users], dtype=np.float32)

    # Mean-centre each rating before building the matrix
    df = df.copy()
    df["rating_centred"] = df["rating"] - df["userId"].map(user_mean_series)

    # Build sparse matrix from (row, col, value) triplets
    rows = df["userId"].map(user_index).values
    cols = df["movieId"].map(item_index).values
    vals = df["rating_centred"].values.astype(np.float32)

    R = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    print(f"Sparse matrix: {R.nnz:,} non-zero entries  ({R.nnz * 4 / 1e6:.0f} MB)")

    return R, user_index, item_index, index_to_item, user_means


def fit(
    ratings_path: str,
    k: int = 50,
    min_ratings_per_user: int = 50,
) -> SVDModel:
    """Fit SVD collaborative filter."""
    df = load_data(ratings_path, min_ratings_per_user)
    R, user_index, item_index, index_to_item, user_means = build_sparse_matrix(df)

    mean_rating = df["rating"].mean()

    print(f"Running SVD (k={k})... ")
    # svds computes ONLY top-k — never materialises full decomposition
    # Returns singular values in ascending order — reverse to get descending

    U, s, Vt = svds(R, k=k)
    U = U[:, ::-1].astype(np.float32)
    s = s[::-1].astype(np.float32)
    Vt = Vt[::-1, :].astype(np.float32)

    print("SVD complete.")
    return SVDModel(
        U=U,
        S=s,
        Vt=Vt,
        user_index=user_index,
        item_index=item_index,
        index_to_item=index_to_item,
        mean_rating=mean_rating,
        user_means=user_means,
    )


def predict_ratings(model: SVDModel, user_id: int) -> np.ndarray:
    """Predict ratings for all items for a given user."""
    if user_id not in model.user_index:
        raise ValueError(f"Unknown user_id: {user_id}")

    u_idx = model.user_index[user_id]
    user_vec = model.U[u_idx]  # (k,)

    # Reconstruct ratings: u · Σ · Vt
    predicted = (user_vec * model.S) @ model.Vt  # (n_items,)

    # Add user mean back
    predicted += model.user_means[u_idx]
    return predicted


def recommend(
    model: SVDModel,
    user_id: int,
    movies_df: pd.DataFrame,
    n: int = 10,
    already_rated: set | None = None,
) -> list[dict]:
    """Return top-n recommendations for a user."""
    scores = predict_ratings(model, user_id)

    # Exclude already-rated movies
    if already_rated:
        for movie_id in already_rated:
            if movie_id in model.item_index:
                scores[model.item_index[movie_id]] = -np.inf

    top_indices = np.argsort(scores)[::-1][:n]

    results = []
    for idx in top_indices:
        movie_id = model.index_to_item[idx]
        movie_row = movies_df[movies_df["movieId"] == movie_id]
        title = movie_row["title"].values[0] if len(movie_row) else f"Movie {movie_id}"
        genres = movie_row["genres"].values[0] if len(movie_row) else "Unknown"
        results.append(
            {
                "movieId": int(movie_id),
                "title": title,
                "genres": genres,
                "predicted_rating": round(float(scores[idx]), 2),
            }
        )
    return results
