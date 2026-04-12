import pickle
import json
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from src.svd_cf import recommend, similar_movies, popular_movies

# ── App state ─────────────────────────────────────────────────────────────
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and data once at startup
    print("Loading SVD model...")
    with open("svd_model.pkl", "rb") as f:
        state["model"] = pickle.load(f)

    state["movies"] = pd.read_csv("ml-25m/movies.csv")
    ratings_df = pd.read_csv("ml-25m/ratings.csv", usecols=["movieId", "rating"])
    state["ratings"] = ratings_df
    state["rating_counts"] = ratings_df["movieId"].value_counts().to_dict()

    # Redis connection
    try:
        state["redis"] = redis.Redis(host="redis", port=6379, decode_responses=True)
        state["redis"].ping()
        print("Redis connected")
    except Exception:
        print("Redis unavailable — running without cache")
        state["redis"] = None

    yield
    state.clear()


app = FastAPI(title="SVD Recommender", lifespan=lifespan)


@app.get("/recommend")
def get_recommendations(
    user_id: int = Query(..., description="MovieLens user ID"),
    n: int = Query(10, description="Number of recommendations", ge=1, le=50),
):
    model = state["model"]
    movies = state["movies"]
    cache = state.get("redis")

    if user_id not in model.user_index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    cache_key = f"rec:{user_id}:{n}"

    # Check Redis cache first
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return {
                "user_id": user_id,
                "source": "cache",
                "recommendations": json.loads(cached),
            }

    # Compute recommendations
    recs = recommend(model, user_id, movies, n=n)

    # Store in Redis with 1-hour TTL
    if cache:
        cache.setex(cache_key, 3600, json.dumps(recs))

    return {"user_id": user_id, "source": "computed", "recommendations": recs}


@app.get("/similar-movies")
def get_similar_movies(
    movie_id: int = Query(..., description="MovieLens movie ID"),
    n: int = Query(10, description="Number of similar movies", ge=1, le=50),
):
    model = state["model"]
    movies = state["movies"]
    cache = state.get("redis")

    if movie_id not in model.item_index:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    cache_key = f"sim:{movie_id}:{n}"

    if cache:
        cached = cache.get(cache_key)
        if cached:
            return {
                "movie_id": movie_id,
                "source": "cache",
                "similar": json.loads(cached),
            }

    results = similar_movies(
        model,
        movie_id,
        movies,
        n=n,
        min_ratings=500,
        rating_counts=state.get("rating_counts"),
    )

    if cache:
        cache.setex(cache_key, 3600, json.dumps(results))

    return {"movie_id": movie_id, "source": "computed", "similar": results}


@app.get("/popular")
def get_popular(
    n: int = Query(10, description="Number of results", ge=1, le=50),
    genre: str | None = Query(None, description="Filter by genre e.g. Action, Drama"),
):
    movies = state["movies"]
    ratings = state["ratings"]
    cache = state.get("redis")

    # Cache key includes genre so different filters are cached separately
    cache_key = f"popular:{n}:{genre or 'all'}"

    if cache:
        cached = cache.get(cache_key)
        if cached:
            return {"source": "cache", "genre": genre, "results": json.loads(cached)}

    results = popular_movies(ratings, movies, n=n, genre=genre)

    if cache:
        # 24-hour TTL — popularity changes slowly
        cache.setex(cache_key, 86400, json.dumps(results))

    return {"source": "computed", "genre": genre, "results": results}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}
