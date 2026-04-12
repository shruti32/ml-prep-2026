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
    _df: pd.DataFrame | None = None,
) -> SVDModel:
    """Fit SVD collaborative filter."""
    # Use provided df instead of loading from disk if given
    df = _df if _df is not None else load_data(ratings_path, min_ratings_per_user)
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


def train_test_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test per user.
    Each user contributes test_frac of their ratings to the test set.
    This ensures every user appears in both splits.
    """
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []

    for _, user_df in df.groupby("userId"):
        idx = user_df.index.tolist()
        rng.shuffle(idx)
        split = int(len(idx) * (1 - test_frac))
        train_rows.extend(idx[:split])
        test_rows.extend(idx[split:])

    return df.loc[train_rows].reset_index(drop=True), df.loc[test_rows].reset_index(
        drop=True
    )


def evaluate_rmse(model: SVDModel, test_df: pd.DataFrame) -> float:
    """
    Compute RMSE on held-out ratings.
    Skips (user, movie) pairs not seen during training.
    """
    errors = []

    for row in test_df.itertuples(index=False):
        # Skip if user or movie wasn't in the training set
        if row.userId not in model.user_index:
            continue
        if row.movieId not in model.item_index:
            continue

        u_idx = model.user_index[row.userId]
        i_idx = model.item_index[row.movieId]

        # Predicted rating: reconstruct from latent factors + user mean
        predicted = (
            float((model.U[u_idx] * model.S) @ model.Vt[:, i_idx])
            + model.user_means[u_idx]
        )

        errors.append((predicted - row.rating) ** 2)

    rmse = float(np.sqrt(np.mean(errors)))
    return rmse


def similar_movies(
    model: SVDModel,
    movie_id: int,
    movies_df: pd.DataFrame,
    n: int = 10,
    min_ratings: int = 500,
    rating_counts: dict | None = None,
) -> list[dict]:
    """
    Find top-n movies most similar to movie_id using cosine similarity
    on item latent vectors (columns of Vt).
    """
    if movie_id not in model.item_index:
        raise ValueError(f"Unknown movie_id: {movie_id}")

    idx = model.item_index[movie_id]

    # Each movie is a column in Vt — shape (k, n_items)
    # Transpose to get (n_items, k) so each row is one movie vector
    item_vectors = model.Vt.T  # (n_items, k)

    target_vec = item_vectors[idx]  # (k,)

    # Cosine similarity: dot product of unit vectors
    norms = np.linalg.norm(item_vectors, axis=1)  # (n_items,)
    target_norm = np.linalg.norm(target_vec)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        similarities = (item_vectors @ target_vec) / (norms * target_norm)
        similarities = np.nan_to_num(similarities)

    # Exclude the query movie itself
    similarities[idx] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:n]

    results = []
    for i in top_indices:
        mid = model.index_to_item[i]
        if rating_counts and rating_counts.get(mid, 0) < min_ratings:
            continue
        if len(results) >= n:
            break
        movie_row = movies_df[movies_df["movieId"] == mid]
        title = movie_row["title"].values[0] if len(movie_row) else f"Movie {mid}"
        genres = movie_row["genres"].values[0] if len(movie_row) else "Unknown"
        results.append(
            {
                "movieId": int(mid),
                "title": title,
                "genres": genres,
                "similarity": round(float(similarities[i]), 4),
            }
        )

    return results


def popular_movies(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10,
    genre: str | None = None,
    min_ratings: int = 100,
) -> list[dict]:
    """
    Return top-n popular movies using Bayesian average.
    Optionally filter by genre.

    Bayesian average formula:
        score = (v * r_bar + m * C) / (v + m)

    Where:
        v     = number of ratings for this movie
        r_bar = mean rating for this movie
        m     = minimum ratings threshold (dampening factor)
        C     = global mean rating across all movies
    """
    # Global mean across all ratings
    C = float(ratings_df["rating"].mean())
    m = min_ratings

    # Per-movie stats
    stats = (
        ratings_df.groupby("movieId")["rating"]
        .agg(rating_count="count", mean_rating="mean")
        .reset_index()
    )

    # Bayesian average
    stats["score"] = (stats["rating_count"] * stats["mean_rating"] + m * C) / (
        stats["rating_count"] + m
    )

    # Merge with movie metadata
    merged = stats.merge(movies_df, on="movieId")

    # Filter by genre if provided
    if genre:
        merged = merged[merged["genres"].str.contains(genre, case=False, na=False)]

    # Filter out movies with too few ratings
    merged = merged[merged["rating_count"] >= min_ratings]

    top = merged.nlargest(n, "score")

    return [
        {
            "movieId": int(row.movieId),
            "title": row.title,
            "genres": row.genres,
            "rating_count": int(row.rating_count),
            "mean_rating": round(float(row.mean_rating), 3),
            "bayesian_score": round(float(row.score), 4),
        }
        for row in top.itertuples()
    ]
