import pickle
import json
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from src.svd_cf import recommend

# ── App state ─────────────────────────────────────────────────────────────
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and data once at startup
    print("Loading SVD model...")
    with open("svd_model.pkl", "rb") as f:
        state["model"] = pickle.load(f)

    state["movies"] = pd.read_csv("ml-25m/movies.csv")

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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}
