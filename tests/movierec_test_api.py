import json
import pytest


# ── /health ───────────────────────────────────────────────────────────────


async def test_health_returns_200(client):
    c, _ = client
    response = await c.get("/health")
    assert response.status_code == 200


async def test_health_status_ok(client):
    c, _ = client
    data = response = (await c.get("/health")).json()
    assert data["status"] == "ok"


async def test_health_model_loaded(client):
    c, _ = client
    data = (await c.get("/health")).json()
    assert data["model_loaded"] is True


# ── /recommend — happy path ───────────────────────────────────────────────


async def test_recommend_valid_user_returns_200(client):
    c, _ = client
    response = await c.get("/recommend?user_id=1")
    assert response.status_code == 200


async def test_recommend_returns_list(client):
    c, _ = client
    data = (await c.get("/recommend?user_id=1")).json()
    assert isinstance(data["recommendations"], list)


async def test_recommend_default_n_is_10(client):
    c, _ = client
    data = (await c.get("/recommend?user_id=1")).json()
    # synthetic model only has 10 movies so result may be <= 10
    assert len(data["recommendations"]) <= 10


async def test_recommend_n_parameter_respected(client):
    c, _ = client
    data = (await c.get("/recommend?user_id=1&n=3")).json()
    assert len(data["recommendations"]) == 3


async def test_recommend_result_has_required_fields(client):
    c, _ = client
    data = (await c.get("/recommend?user_id=1")).json()
    rec = data["recommendations"][0]
    assert "movieId" in rec
    assert "title" in rec
    assert "genres" in rec
    assert "predicted_rating" in rec


async def test_recommend_user_id_in_response(client):
    c, _ = client
    data = (await c.get("/recommend?user_id=1")).json()
    assert data["user_id"] == 1


# ── /recommend — error cases ──────────────────────────────────────────────


async def test_recommend_unknown_user_returns_404(client):
    c, _ = client
    response = await c.get("/recommend?user_id=99999")
    assert response.status_code == 404


async def test_recommend_unknown_user_error_message(client):
    c, _ = client
    data = (await c.get("/recommend?user_id=99999")).json()
    assert "99999" in data["detail"]


async def test_recommend_missing_user_id_returns_422(client):
    """FastAPI returns 422 Unprocessable Entity when a required param is missing."""
    c, _ = client
    response = await c.get("/recommend")
    assert response.status_code == 422


async def test_recommend_n_too_large_returns_422(client):
    """n is capped at 50 in the Query definition."""
    c, _ = client
    response = await c.get("/recommend?user_id=1&n=999")
    assert response.status_code == 422


# ── Redis caching ─────────────────────────────────────────────────────────


async def test_first_request_source_is_computed(client):
    c, fake_redis = client
    fake_redis.flushall()
    data = (await c.get("/recommend?user_id=1&n=3")).json()
    assert data["source"] == "computed"


async def test_second_request_source_is_cache(client):
    c, fake_redis = client
    fake_redis.flushall()
    await c.get("/recommend?user_id=1&n=3")  # populates cache
    data = (await c.get("/recommend?user_id=1&n=3")).json()  # hits cache
    assert data["source"] == "cache"


async def test_cache_returns_same_recommendations(client):
    c, fake_redis = client
    fake_redis.flushall()
    first = (await c.get("/recommend?user_id=1&n=3")).json()["recommendations"]
    second = (await c.get("/recommend?user_id=1&n=3")).json()["recommendations"]
    assert first == second


async def test_different_users_have_separate_cache_keys(client):
    c, fake_redis = client
    fake_redis.flushall()
    await c.get("/recommend?user_id=1&n=3")
    data = (await c.get("/recommend?user_id=2&n=3")).json()
    # User 2's first request should still be computed, not served from user 1's cache
    assert data["source"] == "computed"
