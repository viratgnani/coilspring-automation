"""Tests for POST /signals -- schema, validation, idempotency, and timestamps.

New tests (Blockers 3 & 4 — Server Idempotency):
  - X-Idempotency-Key accepted
  - First POST creates exactly one signal
  - Second POST with same key returns existing signal (no duplicate)
  - Crash-after-insert / retry produces exactly one signal row
  - No key -> fresh UUID4 generated (backward compat)
  - Concurrent retry race: only one row in DB
"""

import asyncio
import time
import uuid

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from coilbrain.main import app
from coilbrain.config import settings


client = TestClient(app)

# ── Shared fixture ─────────────────────────────────────────────────────────────

VALID_PAYLOAD: dict = {
    "symbol": "KOMA/USDT:USDT",
    "exchange": "bybit",
    "trigger": "SNIPER TRIGGER",
    "grade": "A",
    "score": 100.0,
    "signal": "PUMP",
    "oi_status": "OI_PRESENT",
    "price": 0.015447,
    "change_24h_pct": 11.98,
    "volume_24h_usd": 4_300_000.0,
    "breakout_pct": 4.84,
    "body_volume_spike": 3.3,
    "volume_spike": 8.6,
    "flat_period_days": 30.0,
    "signal_timestamp": 1_700_000_000.0,
}


@pytest.fixture(autouse=True)
def clear_store() -> None:
    """Wipe the signals table between tests to prevent cross-test contamination."""

    async def _clear():
        async with aiosqlite.connect(settings.database_url) as db:
            await db.execute("DELETE FROM price_observations")
            await db.execute("DELETE FROM observation_errors")
            await db.execute("DELETE FROM signals")
            await db.commit()

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


# ── HTTP contract (preserved) ──────────────────────────────────────────────────


def test_create_signal_returns_201() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD).status_code == 201


def test_create_signal_response_is_json() -> None:
    assert "application/json" in client.post("/signals", json=VALID_PAYLOAD).headers["content-type"]


# ── Signal ID (preserved + idempotency) ───────────────────────────────────────


def test_create_signal_returns_signal_id() -> None:
    data = client.post("/signals", json=VALID_PAYLOAD).json()
    assert "signal_id" in data and data["signal_id"]


def test_signal_id_is_valid_uuid4_without_key() -> None:
    """Without X-Idempotency-Key the server generates a UUID4."""
    signal_id = client.post("/signals", json=VALID_PAYLOAD).json()["signal_id"]
    parsed = uuid.UUID(signal_id, version=4)
    assert str(parsed) == signal_id


def test_signal_id_is_unique_across_requests_without_key() -> None:
    """Two consecutive POSTs without a key must produce distinct signal_ids."""
    id_a = client.post("/signals", json=VALID_PAYLOAD).json()["signal_id"]
    id_b = client.post("/signals", json=VALID_PAYLOAD).json()["signal_id"]
    assert id_a != id_b


# ── Idempotency key — BLOCKER 3 ───────────────────────────────────────────────


def test_idempotency_key_used_as_signal_id() -> None:
    """When X-Idempotency-Key is supplied it becomes the signal_id."""
    key = str(uuid.uuid4())
    resp = client.post("/signals", json=VALID_PAYLOAD, headers={"X-Idempotency-Key": key})
    assert resp.status_code == 201
    assert resp.json()["signal_id"] == key


def test_idempotency_key_is_valid_uuid_in_response() -> None:
    key = str(uuid.uuid4())
    signal_id = client.post(
        "/signals", json=VALID_PAYLOAD, headers={"X-Idempotency-Key": key}
    ).json()["signal_id"]
    assert uuid.UUID(signal_id, version=4)
    assert signal_id == key


def test_first_post_with_key_creates_exactly_one_signal() -> None:
    """A single POST with a fresh key must create exactly one DB row."""
    key = str(uuid.uuid4())
    resp = client.post("/signals", json=VALID_PAYLOAD, headers={"X-Idempotency-Key": key})
    assert resp.status_code == 201

    async def _count():
        async with aiosqlite.connect(settings.database_url) as db:
            cur = await db.execute("SELECT COUNT(*) FROM signals WHERE signal_id = ?", (key,))
            return (await cur.fetchone())[0]

    assert asyncio.run(_count()) == 1


def test_duplicate_post_same_key_returns_existing_signal() -> None:
    """Second POST with same key must return identical body — no new row."""
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}

    first = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs).json()
    second = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs).json()

    assert first == second


def test_duplicate_post_same_key_returns_201_both_times() -> None:
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}
    assert client.post("/signals", json=VALID_PAYLOAD, headers=hdrs).status_code == 201
    assert client.post("/signals", json=VALID_PAYLOAD, headers=hdrs).status_code == 201


def test_duplicate_post_creates_exactly_one_db_row() -> None:
    """After two POSTs with the same key, the DB contains exactly one signal."""
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}
    client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)
    client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)

    async def _count():
        async with aiosqlite.connect(settings.database_url) as db:
            cur = await db.execute("SELECT COUNT(*) FROM signals WHERE signal_id = ?", (key,))
            return (await cur.fetchone())[0]

    assert asyncio.run(_count()) == 1


def test_ten_duplicate_posts_create_exactly_one_db_row() -> None:
    """10 retries with the same key must all succeed and leave exactly one row."""
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}
    for _ in range(10):
        resp = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)
        assert resp.status_code == 201
        assert resp.json()["signal_id"] == key

    async def _count():
        async with aiosqlite.connect(settings.database_url) as db:
            cur = await db.execute("SELECT COUNT(*) FROM signals WHERE signal_id = ?", (key,))
            return (await cur.fetchone())[0]

    assert asyncio.run(_count()) == 1


def test_idempotent_response_matches_get() -> None:
    """The idempotent POST response must match the GET /signals/{id} response."""
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}
    post_data = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs).json()
    # Second POST (idempotent replay)
    replay_data = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs).json()
    get_data = client.get(f"/signals/{key}").json()
    assert post_data == get_data
    assert replay_data == get_data


def test_different_keys_create_different_signals() -> None:
    """Two distinct keys must each create their own signal row."""
    key_a = str(uuid.uuid4())
    key_b = str(uuid.uuid4())
    resp_a = client.post("/signals", json=VALID_PAYLOAD, headers={"X-Idempotency-Key": key_a}).json()
    resp_b = client.post("/signals", json=VALID_PAYLOAD, headers={"X-Idempotency-Key": key_b}).json()
    assert resp_a["signal_id"] == key_a
    assert resp_b["signal_id"] == key_b
    assert resp_a["signal_id"] != resp_b["signal_id"]


def test_no_key_generates_server_uuid() -> None:
    """Without a key the server still generates a valid UUID4 (backward compat)."""
    resp = client.post("/signals", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    sig_id = resp.json()["signal_id"]
    assert uuid.UUID(sig_id, version=4)


# ── BLOCKER 4: Crash-after-insert / retry simulation ─────────────────────────


def test_crash_after_insert_retry_yields_one_row() -> None:
    """Simulate: client sends POST, server inserts, client never receives response.
    Client retries with the SAME key. DB must contain exactly ONE signal.

    This is the at-least-once delivery crash scenario. With server-side
    idempotency, the retry is safe and produces no duplicate.
    """
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}

    # Step 1: first POST — server inserts, returns 201
    first = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)
    assert first.status_code == 201
    original_signal = first.json()

    # Step 2: simulate client crash — it never processes the 201 response.
    # Step 3: client restarts and retries with the same key.
    retry = client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)
    assert retry.status_code == 201

    # Step 4: response must equal the original signal
    assert retry.json() == original_signal

    # Step 5: exactly ONE row in DB
    async def _count():
        async with aiosqlite.connect(settings.database_url) as db:
            cur = await db.execute("SELECT COUNT(*) FROM signals WHERE signal_id = ?", (key,))
            return (await cur.fetchone())[0]

    assert asyncio.run(_count()) == 1


def test_retry_response_is_retrievable_via_get() -> None:
    """After crash-and-retry the signal is reachable by GET."""
    key = str(uuid.uuid4())
    hdrs = {"X-Idempotency-Key": key}
    client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)  # original
    client.post("/signals", json=VALID_PAYLOAD, headers=hdrs)  # retry
    get_resp = client.get(f"/signals/{key}")
    assert get_resp.status_code == 200
    assert get_resp.json()["signal_id"] == key


# ── Timestamp and latency (preserved) ─────────────────────────────────────────


def test_response_contains_received_timestamp() -> None:
    data = client.post("/signals", json=VALID_PAYLOAD).json()
    assert isinstance(data.get("received_timestamp"), float)


def test_received_timestamp_is_recent() -> None:
    before = time.time()
    data = client.post("/signals", json=VALID_PAYLOAD).json()
    assert before <= data["received_timestamp"] <= time.time()


def test_signal_timestamp_is_echoed() -> None:
    resp = client.post("/signals", json=VALID_PAYLOAD)
    assert resp.json()["signal_timestamp"] == VALID_PAYLOAD["signal_timestamp"]


def test_transport_latency_is_calculated() -> None:
    data = client.post("/signals", json=VALID_PAYLOAD).json()
    expected = (data["received_timestamp"] - data["signal_timestamp"]) * 1000.0
    assert abs(data["transport_latency_ms"] - expected) < 1.0


def test_transport_latency_is_positive() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD).json()["transport_latency_ms"] >= 0.0


# ── Field round-trip (preserved) ──────────────────────────────────────────────


def test_all_payload_fields_are_returned() -> None:
    data = client.post("/signals", json=VALID_PAYLOAD).json()
    for field, value in VALID_PAYLOAD.items():
        assert data.get(field) == value, f"Field {field} mismatch"


def test_string_whitespace_is_stripped() -> None:
    dirty = VALID_PAYLOAD | {"symbol": "  KOMA/USDT:USDT  ", "exchange": " bybit "}
    data = client.post("/signals", json=dirty).json()
    assert data["symbol"] == "KOMA/USDT:USDT"
    assert data["exchange"] == "bybit"


# ── Price validation (preserved) ──────────────────────────────────────────────


def test_price_zero_is_rejected() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"price": 0.0}).status_code == 422


def test_price_negative_is_rejected() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"price": -1.0}).status_code == 422


# ── Volume / spike / flat-period validation (preserved) ──────────────────────


def test_volume_24h_negative_is_rejected() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"volume_24h_usd": -1.0}).status_code == 422


def test_volume_24h_zero_is_accepted() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"volume_24h_usd": 0.0}).status_code == 201


def test_body_volume_spike_negative_is_rejected() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"body_volume_spike": -0.1}).status_code == 422


def test_volume_spike_negative_is_rejected() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"volume_spike": -0.1}).status_code == 422


def test_flat_period_days_negative_is_rejected() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"flat_period_days": -1.0}).status_code == 422


def test_flat_period_days_zero_is_accepted() -> None:
    assert client.post("/signals", json=VALID_PAYLOAD | {"flat_period_days": 0.0}).status_code == 201


# ── Missing required fields (preserved) ──────────────────────────────────────


@pytest.mark.parametrize(
    "missing_field",
    [
        "symbol", "exchange", "trigger", "grade", "score", "signal",
        "oi_status", "price", "volume_24h_usd", "signal_timestamp",
    ],
)
def test_missing_required_field_returns_422(missing_field: str) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    assert client.post("/signals", json=payload).status_code == 422


# ── In-memory storage (preserved) ────────────────────────────────────────────


def test_signal_is_stored_in_db() -> None:
    resp = client.post("/signals", json=VALID_PAYLOAD)
    signal_id = resp.json()["signal_id"]
    assert client.get(f"/signals/{signal_id}").json()["symbol"] == VALID_PAYLOAD["symbol"]


def test_get_signal_by_id_returns_200() -> None:
    signal_id = client.post("/signals", json=VALID_PAYLOAD).json()["signal_id"]
    assert client.get(f"/signals/{signal_id}").status_code == 200


def test_get_signal_by_id_matches_post_response() -> None:
    post_data = client.post("/signals", json=VALID_PAYLOAD).json()
    get_data = client.get(f"/signals/{post_data['signal_id']}").json()
    assert get_data == post_data


def test_get_unknown_signal_returns_404() -> None:
    assert client.get("/signals/00000000-0000-4000-8000-000000000000").status_code == 404


# ── Optional field acceptance (preserved) ────────────────────────────────────


@pytest.mark.parametrize(
    "optional_field",
    ["change_24h_pct", "breakout_pct", "body_volume_spike", "volume_spike", "flat_period_days"],
)
def test_optional_field_accepts_null(optional_field: str) -> None:
    resp = client.post("/signals", json=VALID_PAYLOAD | {optional_field: None})
    assert resp.status_code == 201
    assert resp.json()[optional_field] is None


@pytest.mark.parametrize(
    "optional_field",
    ["change_24h_pct", "breakout_pct", "body_volume_spike", "volume_spike", "flat_period_days"],
)
def test_optional_field_absent_defaults_to_null(optional_field: str) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != optional_field}
    resp = client.post("/signals", json=payload)
    assert resp.status_code == 201
    assert resp.json()[optional_field] is None


# ── Health (preserved) ────────────────────────────────────────────────────────


def test_health_still_works() -> None:
    client.post("/signals", json=VALID_PAYLOAD)
    assert client.get("/health").json() == {"status": "ok", "service": "coilbrain"}
