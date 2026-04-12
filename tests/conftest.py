import json
import pickle
from unittest.mock import MagicMock, mock_open, patch
from contextlib import asynccontextmanager
from src.movie_rec_api import state, app

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.movie_rec_api import app
from src.svd_cf import SVDModel


# ── Synthetic model ───────────────────────────────────────────────────────
@pytest.fixture
def synthetic_model():
    """
    A tiny SVDModel with 5 users, 10 movies, k=3.
    Fast to build, no file I/O needed.
    """
    rng = np.random.default_rng(42)
    n_users, n_items, k = 5, 10, 3

    user_ids = [1, 2, 3, 4, 5]
    movie_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    return SVDModel(
        U=rng.standard_normal((n_users, k)).astype(np.float32),
        S=np.array([5.0, 3.0, 1.0], dtype=np.float32),
        Vt=rng.standard_normal((k, n_items)).astype(np.float32),
        user_index={uid: i for i, uid in enumerate(user_ids)},
        item_index={mid: j for j, mid in enumerate(movie_ids)},
        index_to_item={j: mid for j, mid in enumerate(movie_ids)},
        mean_rating=3.5,
        user_means=rng.uniform(3.0, 4.5, n_users).astype(np.float32),
    )


# ── Synthetic movies dataframe ────────────────────────────────────────────
@pytest.fixture
def mock_movies():
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "title": [f"Movie {i}" for i in range(1, 11)],
            "genres": ["Action|Drama"] * 10,
        }
    )


# ── In-memory mock Redis ──────────────────────────────────────────────────
class FakeRedis:
    """
    Behaves like a real Redis client but stores data in a dict.
    Lets us test caching logic without a running Redis instance.
    """

    def __init__(self):
        self._store = {}

    def ping(self):
        return True

    def get(self, key):
        return self._store.get(key)

    def setex(self, key, ttl, value):
        self._store[key] = value

    def flushall(self):
        self._store.clear()


# ── Test client ───────────────────────────────────────────────────────────
@pytest.fixture
async def client(synthetic_model, mock_movies):
    fake_redis = FakeRedis()

    # Replace the app's lifespan with a no-op
    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = mock_lifespan

    # Populate state directly — no file I/O needed
    state.clear()
    state["model"] = synthetic_model
    state["movies"] = mock_movies
    state["redis"] = fake_redis

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c, fake_redis

    # Restore everything after the test
    app.router.lifespan_context = original_lifespan
    state.clear()
