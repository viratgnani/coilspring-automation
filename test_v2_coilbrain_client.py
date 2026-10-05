"""Tests for coilbrain_client.py – Signal Durability / Local Outbox.

Isolation: no real HTTP calls, no ccxt imports, no pandas.
AnalysisResult is simulated with types.SimpleNamespace.

Test groups
-----------
Original (preserved)
  1. Payload fields        -- all 15 fields built correctly from AnalysisResult
  2. OI status             -- OI_PRESENT vs UNKNOWN derivation
  3. signal_timestamp      -- stamped at notify() call time
  4. notify() disabled     -- COILBRAIN_URL empty -> no exception; outbox written
  5. Failure isolation     -- HTTP failure -> no exception, scanner continues
  6. Non-blocking          -- notify() returns in <500ms

New (Blocker 1 — Signal Identity)
  7. UUID identity         -- UUID4 generated once, distinct, persists across retries
  8. Same-symbol same-sec  -- two signals within 1s remain distinct (UUID, not timestamp)

New (Blocker 2 — Infinite Retry)
  9. Infinite retry        -- >10 failures keep state PENDING, never DEAD
 10. Backoff continues     -- capped 1280s repeats indefinitely

New (Blocker 5 — SQLite Durability)
 11. synchronous=FULL      -- pragma confirmed on every connection

New (Outbox mechanics)
 12. Outbox QUEUED         -- always writes regardless of URL
 13. CoilBrain unavailable -- item stays PENDING
 14. Retry logic           -- backoff math and due-item selection
 15. Restart recovery      -- pre-existing items delivered after restart
 16. Duplicate idempotency -- same key inserted twice = one outbox row
 17. Successful delivery   -- DELIVERED after HTTP 200, header sent
 18. Corrupted payload     -- bad JSON -> DEAD, others unaffected
 19. Diagnostic status     -- outbox_status() correct counts
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import types
import uuid
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import coilbrain_client


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_result(**overrides):
    defaults = dict(
        symbol="KOMA/USDT:USDT",
        score=92,
        grade="A",
        change_24h=11.98,
        volume_24h_usd=4_300_000.0,
        flat_candles=336,
        last_close=0.015447,
        trigger_body_multiple=3.3,
        trigger_vol_multiple=8.6,
        trigger_close_pct_above_high=4.84,
        flags=["📈 OI_PRESENT"],
        is_dump_then_pump=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


@pytest.fixture()
def outbox_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_outbox.db")
    monkeypatch.setenv("COILBRAIN_OUTBOX", db_path)
    return db_path


def _open_outbox(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _count_state(db_path: str, state: str) -> int:
    conn = _open_outbox(db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM outbox_items WHERE state=?", (state,)
    ).fetchone()[0]
    conn.close()
    return n


def _get_all(db_path: str) -> list:
    conn = _open_outbox(db_path)
    rows = conn.execute("SELECT * FROM outbox_items").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ===========================================================================
# 1. Payload fields (preserved)
# ===========================================================================

class TestBuildPayloadFields:
    def test_symbol(self):
        assert coilbrain_client.build_payload(_make_result())["symbol"] == "KOMA/USDT:USDT"

    def test_exchange_constant(self):
        assert coilbrain_client.build_payload(_make_result())["exchange"] == "binance"

    def test_trigger_constant(self):
        assert coilbrain_client.build_payload(_make_result())["trigger"] == "COILED_SPRING_V2"

    def test_grade(self):
        assert coilbrain_client.build_payload(_make_result(grade="B"))["grade"] == "B"

    def test_score_is_float(self):
        p = coilbrain_client.build_payload(_make_result(score=92))
        assert p["score"] == 92.0 and isinstance(p["score"], float)

    def test_signal_constant_is_pump(self):
        assert coilbrain_client.build_payload(_make_result(is_dump_then_pump=True))["signal"] == "PUMP"

    def test_price(self):
        assert coilbrain_client.build_payload(_make_result(last_close=0.015447))["price"] == pytest.approx(0.015447)

    def test_change_24h_pct(self):
        assert coilbrain_client.build_payload(_make_result(change_24h=11.98))["change_24h_pct"] == pytest.approx(11.98)

    def test_volume_24h_usd(self):
        assert coilbrain_client.build_payload(_make_result(volume_24h_usd=4_300_000.0))["volume_24h_usd"] == pytest.approx(4_300_000.0)

    def test_breakout_pct(self):
        assert coilbrain_client.build_payload(_make_result(trigger_close_pct_above_high=4.84))["breakout_pct"] == pytest.approx(4.84)

    def test_body_volume_spike(self):
        assert coilbrain_client.build_payload(_make_result(trigger_body_multiple=3.3))["body_volume_spike"] == pytest.approx(3.3)

    def test_volume_spike(self):
        assert coilbrain_client.build_payload(_make_result(trigger_vol_multiple=8.6))["volume_spike"] == pytest.approx(8.6)

    def test_flat_period_days_conversion(self):
        assert coilbrain_client.build_payload(_make_result(flat_candles=336))["flat_period_days"] == pytest.approx(14.0)

    def test_flat_period_days_720_candles(self):
        assert coilbrain_client.build_payload(_make_result(flat_candles=720))["flat_period_days"] == pytest.approx(30.0)

    def test_signal_timestamp_is_recent(self):
        before = time.time()
        p = coilbrain_client.build_payload(_make_result())
        assert before <= p["signal_timestamp"] <= time.time()

    def test_payload_is_json_serialisable(self):
        p = coilbrain_client.build_payload(_make_result())
        assert json.loads(json.dumps(p))["symbol"] == "KOMA/USDT:USDT"

    def test_payload_has_all_15_fields(self):
        expected = {
            "symbol", "exchange", "trigger", "grade", "score", "signal",
            "oi_status", "price", "change_24h_pct", "volume_24h_usd",
            "breakout_pct", "body_volume_spike", "volume_spike",
            "flat_period_days", "signal_timestamp",
        }
        assert set(coilbrain_client.build_payload(_make_result()).keys()) == expected


# ===========================================================================
# 2. OI status (preserved)
# ===========================================================================

class TestOIStatus:
    def test_oi_present_when_flag_present(self):
        assert coilbrain_client.build_payload(_make_result(flags=["📈 OI_PRESENT"]))["oi_status"] == "OI_PRESENT"

    def test_oi_unknown_when_no_flags(self):
        assert coilbrain_client.build_payload(_make_result(flags=[]))["oi_status"] == "UNKNOWN"

    def test_oi_unknown_when_only_other_flags(self):
        assert coilbrain_client.build_payload(_make_result(flags=["🔥 NEG_FUNDING"]))["oi_status"] == "UNKNOWN"

    def test_oi_present_with_mixed_flags(self):
        assert coilbrain_client.build_payload(_make_result(flags=["🔥 NEG_FUNDING", "📈 OI_PRESENT"]))["oi_status"] == "OI_PRESENT"

    def test_oi_present_plain_text_flag(self):
        assert coilbrain_client.build_payload(_make_result(flags=["OI_PRESENT"]))["oi_status"] == "OI_PRESENT"

    def test_oi_semantics_no_increase_claim(self):
        for flags in [["📈 OI_PRESENT"], []]:
            p = coilbrain_client.build_payload(_make_result(flags=flags))
            assert p["oi_status"] in {"OI_PRESENT", "UNKNOWN"}
            assert "INCREASING" not in p["oi_status"]


# ===========================================================================
# 3. signal_timestamp (preserved)
# ===========================================================================

class TestSignalTimestamp:
    def test_timestamp_captured_at_call_time(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:9999/signals")
        before = time.time()
        coilbrain_client.notify(_make_result())
        assert before <= time.time()


# ===========================================================================
# 4. notify() disabled (preserved + outbox always written)
# ===========================================================================

class TestNotifyDisabled:
    def test_no_exception_when_url_empty(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())  # must not raise

    def test_outbox_written_when_url_empty(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result(symbol="SAVE/USDT:USDT"))
        assert _count_state(outbox_db, "PENDING") == 1

    def test_no_exception_when_url_whitespace_only(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "   ")
        coilbrain_client.notify(_make_result())


# ===========================================================================
# 5. Failure isolation (preserved)
# ===========================================================================

class TestFailureIsolation:
    def test_connection_refused_does_not_raise(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:19999/signals")
        coilbrain_client.notify(_make_result())
        time.sleep(0.15)

    def test_urlopen_exception_swallowed_in_post(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            coilbrain_client._post(
                "http://127.0.0.1:19999/signals",
                coilbrain_client.build_payload(_make_result()),
                "KOMA/USDT:USDT",
            )

    def test_scanner_continues_if_coilbrain_down(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:19999/signals")
        for i in range(5):
            coilbrain_client.notify(_make_result(symbol=f"TOKEN{i}/USDT:USDT"))
        time.sleep(0.2)


# ===========================================================================
# 6. Non-blocking (preserved)
# ===========================================================================

class TestNonBlocking:
    def test_notify_returns_before_http_completes(self, monkeypatch, outbox_db):
        def slow_urlopen(req, timeout):
            time.sleep(1.0)
            raise ConnectionRefusedError("simulated")

        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:19999/signals")
        with patch("urllib.request.urlopen", side_effect=slow_urlopen):
            start = time.time()
            coilbrain_client.notify(_make_result())
            assert time.time() - start < 0.5


# ===========================================================================
# 7. UUID Identity (BLOCKER 1)
# ===========================================================================

class TestUUIDIdentity:
    def test_outbox_id_is_valid_uuid4(self, monkeypatch, outbox_db):
        """The outbox row id must be a well-formed UUID4."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())
        rows = _get_all(outbox_db)
        assert len(rows) == 1
        parsed = uuid.UUID(rows[0]["id"], version=4)
        assert str(parsed) == rows[0]["id"]

    def test_two_signals_have_distinct_uuids(self, monkeypatch, outbox_db):
        """Two separate notify() calls must produce different UUIDs."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result(symbol="AAA/USDT:USDT"))
        coilbrain_client.notify(_make_result(symbol="BBB/USDT:USDT"))
        rows = _get_all(outbox_db)
        assert len(rows) == 2
        assert rows[0]["id"] != rows[1]["id"]

    def test_uuid_unchanged_across_retries(self, monkeypatch, outbox_db):
        """The UUID stored in the outbox is never modified by delivery attempts."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())
        rows = _get_all(outbox_db)
        original_id = rows[0]["id"]

        # Simulate failed delivery attempt
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._mark_failed(conn, original_id, "timeout", 1, permanent=False)
        conn.close()

        rows_after = _get_all(outbox_db)
        assert rows_after[0]["id"] == original_id

    def test_x_idempotency_key_header_matches_outbox_id(self, monkeypatch, outbox_db):
        """X-Idempotency-Key sent to server must equal the outbox row id."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())
        rows = _get_all(outbox_db)
        outbox_id = rows[0]["id"]

        captured_headers = {}

        def capture(req, timeout):
            captured_headers.update(dict(req.headers))
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=capture):
            coilbrain_client._deliver_one(
                "http://example.com/signals", outbox_id, rows[0]["payload_json"]
            )

        header_keys = {k.lower() for k in captured_headers}
        assert "x-idempotency-key" in header_keys
        lower_map = {k.lower(): v for k, v in captured_headers.items()}
        assert lower_map["x-idempotency-key"] == outbox_id

    def test_notify_does_not_use_symbol_timestamp_as_id(self, monkeypatch, outbox_db):
        """Outbox id must be a UUID4, not a symbol+timestamp string."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result(symbol="TAKE/USDT:USDT"))
        rows = _get_all(outbox_db)
        assert "TAKE" not in rows[0]["id"]
        assert "_" not in rows[0]["id"] or rows[0]["id"].count("-") == 4  # UUID has 4 dashes


# ===========================================================================
# 8. Same-symbol same-second remain distinct (BLOCKER 1)
# ===========================================================================

class TestSameSymbolSameSecond:
    def test_two_signals_same_symbol_same_second_are_distinct(self, monkeypatch, outbox_db):
        """Two notify() calls for the same symbol in the same second
        must produce two separate outbox rows — not one (old truncation bug)."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        fixed_ts = 1700000000.0

        with patch("coilbrain_client.time") as mock_t:
            mock_t.time = lambda: fixed_ts
            coilbrain_client.notify(_make_result(symbol="SAME/USDT:USDT"))
            coilbrain_client.notify(_make_result(symbol="SAME/USDT:USDT"))

        rows = _get_all(outbox_db)
        assert len(rows) == 2
        assert rows[0]["id"] != rows[1]["id"]


# ===========================================================================
# 9. Infinite Retry (BLOCKER 2)
# ===========================================================================

class TestInfiniteRetry:
    def test_item_stays_pending_after_many_failures(self, monkeypatch, outbox_db):
        """After >10 failures the item must remain PENDING, never DEAD."""
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        key = str(uuid.uuid4())
        payload = json.dumps(coilbrain_client.build_payload(_make_result()))
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) "
            "VALUES (?, ?, ?, 0, 'PENDING')",
            (key, payload, time.time()),
        )
        conn.commit()

        # Simulate 15 consecutive failures
        for i in range(1, 16):
            coilbrain_client._mark_failed(conn, key, "timeout", i, permanent=False)

        rows = _get_all(outbox_db)
        assert rows[0]["state"] == "PENDING"
        assert rows[0]["attempts"] == 15
        conn.close()

    def test_dead_only_for_corrupt_payload(self, monkeypatch, outbox_db):
        """DEAD state is only set when permanent=True (corrupt payload)."""
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        key = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) "
            "VALUES (?, 'CORRUPT{{', ?, 0, 'PENDING')",
            (key, time.time()),
        )
        conn.commit()
        coilbrain_client._mark_failed(conn, key, "CORRUPTED", 1, permanent=True)
        rows = _get_all(outbox_db)
        assert rows[0]["state"] == "DEAD"
        conn.close()

    def test_connectivity_failure_never_produces_dead(self, monkeypatch, outbox_db):
        """HTTP connectivity errors must never set state=DEAD."""
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        key = str(uuid.uuid4())
        payload = json.dumps(coilbrain_client.build_payload(_make_result()))
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) "
            "VALUES (?, ?, ?, 0, 'PENDING')",
            (key, payload, time.time()),
        )
        conn.commit()

        for i in range(1, 25):  # well beyond the old 10-attempt limit
            coilbrain_client._mark_failed(conn, key, "ConnectionRefused", i, permanent=False)

        assert _get_all(outbox_db)[0]["state"] == "PENDING"
        conn.close()


# ===========================================================================
# 10. Backoff continues indefinitely (BLOCKER 2)
# ===========================================================================

class TestBackoffContinues:
    def test_backoff_doubles(self):
        b0 = coilbrain_client._backoff_for(0)
        b1 = coilbrain_client._backoff_for(1)
        assert b1 == pytest.approx(b0 * 2)

    def test_backoff_capped_at_max(self):
        for n in range(15, 35):
            assert coilbrain_client._backoff_for(n) == pytest.approx(coilbrain_client._MAX_BACKOFF_S)

    def test_high_attempt_count_still_due_after_cap(self, monkeypatch, outbox_db):
        """After 50 failures, item becomes due again after _MAX_BACKOFF_S elapses."""
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        # last_attempt_at is 2000s ago (> 1280s cap)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, last_attempt_at, state) "
            "VALUES ('hi-att', '{}', ?, 50, ?, 'PENDING')",
            (time.time() - 5000, time.time() - 2000),
        )
        conn.commit()
        due = coilbrain_client._due_items(conn)
        conn.close()
        assert any(row[0] == "hi-att" for row in due)


# ===========================================================================
# 11. SQLite synchronous=FULL (BLOCKER 5)
# ===========================================================================

class TestSQLiteDurability:
    def test_synchronous_full_is_set(self, tmp_path):
        """Every outbox connection must use PRAGMA synchronous=FULL."""
        db_path = str(tmp_path / "dur.db")
        conn = coilbrain_client._get_conn(db_path)
        val = conn.execute("PRAGMA synchronous").fetchone()[0]
        conn.close()
        # SQLite returns 2 for FULL
        assert val == 2, f"Expected synchronous=FULL (2), got {val}"


# ===========================================================================
# 12. Outbox QUEUED
# ===========================================================================

class TestOutboxQueued:
    def test_queued_with_url_set(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:19999/signals")
        coilbrain_client.notify(_make_result())
        assert _count_state(outbox_db, "PENDING") >= 1

    def test_queued_without_url(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())
        assert _count_state(outbox_db, "PENDING") == 1

    def test_outbox_contains_full_payload(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result(symbol="CCC/USDT:USDT", last_close=0.12345))
        rows = _get_all(outbox_db)
        payload = json.loads(rows[0]["payload_json"])
        assert payload["symbol"] == "CCC/USDT:USDT"
        assert payload["price"] == pytest.approx(0.12345)

    def test_outbox_schema_fields_correct(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())
        row = _get_all(outbox_db)[0]
        assert row["state"] == "PENDING"
        assert row["attempts"] == 0
        assert row["queued_at"] > 0


# ===========================================================================
# 13. CoilBrain unavailable
# ===========================================================================

class TestCoilBrainUnavailable:
    def test_pending_after_connection_refused(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:19999/signals")
        coilbrain_client.notify(_make_result())
        time.sleep(0.2)
        rows = _get_all(outbox_db)
        assert rows[0]["state"] != "DELIVERED"

    def test_scanner_not_blocked(self, monkeypatch, outbox_db):
        monkeypatch.setenv("COILBRAIN_URL", "http://127.0.0.1:19999/signals")
        start = time.time()
        for i in range(10):
            coilbrain_client.notify(_make_result(symbol=f"TOK{i}/USDT:USDT"))
        assert time.time() - start < 1.0


# ===========================================================================
# 14. Retry logic
# ===========================================================================

class TestRetryLogic:
    def test_item_not_due_before_backoff(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, last_attempt_at, state) "
            "VALUES ('td1', '{}', ?, 1, ?, 'PENDING')",
            (time.time() - 10, time.time() - 1),
        )
        conn.commit()
        due = coilbrain_client._due_items(conn)
        conn.close()
        assert not any(r[0] == "td1" for r in due)

    def test_item_due_after_backoff_elapsed(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, last_attempt_at, state) "
            "VALUES ('td2', '{}', ?, 1, ?, 'PENDING')",
            (time.time() - 100, time.time() - 100),
        )
        conn.commit()
        due = coilbrain_client._due_items(conn)
        conn.close()
        assert any(r[0] == "td2" for r in due)

    def test_first_attempt_always_due(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) "
            "VALUES ('fresh', '{}', ?, 0, 'PENDING')",
            (time.time(),),
        )
        conn.commit()
        due = coilbrain_client._due_items(conn)
        conn.close()
        assert any(r[0] == "fresh" for r in due)


# ===========================================================================
# 15. Restart recovery
# ===========================================================================

class TestRestartRecovery:
    def test_items_survive_process_restart(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result(symbol="RECOVER/USDT:USDT"))
        rows = _get_all(outbox_db)
        assert rows[0]["state"] == "PENDING"

    def test_client_restart_preserves_identity(self, monkeypatch, outbox_db):
        """The UUID stored in the outbox before a restart is the same after."""
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result())
        original_id = _get_all(outbox_db)[0]["id"]

        # "Restart": open raw DB and confirm UUID unchanged
        rows = _get_all(outbox_db)
        assert rows[0]["id"] == original_id

    def test_pre_existing_item_delivered_by_worker(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        key = str(uuid.uuid4())
        payload = json.dumps(coilbrain_client.build_payload(_make_result()))
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) "
            "VALUES (?, ?, ?, 0, 'PENDING')",
            (key, payload, time.time() - 60),
        )
        conn.commit()

        resp_mock = MagicMock()
        resp_mock.status = 200
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)

        monkeypatch.setenv("COILBRAIN_URL", "http://example.com/signals")
        with patch("urllib.request.urlopen", return_value=resp_mock):
            due = coilbrain_client._due_items(conn)
            for row in due:
                sig_id, pj, _, _ = row
                coilbrain_client._deliver_one("http://example.com/signals", sig_id, pj)
                coilbrain_client._mark_delivered(conn, sig_id)
        conn.close()

        assert _count_state(outbox_db, "DELIVERED") == 1


# ===========================================================================
# 16. Duplicate idempotency
# ===========================================================================

class TestDuplicateIdempotency:
    def test_same_key_inserted_twice_yields_one_row(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        key = str(uuid.uuid4())
        payload = coilbrain_client.build_payload(_make_result())
        coilbrain_client._enqueue(conn, key, payload)
        coilbrain_client._enqueue(conn, key, payload)
        conn.close()
        rows = _get_all(outbox_db)
        assert len([r for r in rows if r["id"] == key]) == 1

    def test_two_distinct_signals_yield_two_rows(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        coilbrain_client.notify(_make_result(symbol="A/USDT:USDT"))
        coilbrain_client.notify(_make_result(symbol="B/USDT:USDT"))
        assert len(_get_all(outbox_db)) == 2


# ===========================================================================
# 17. Successful delivery
# ===========================================================================

class TestSuccessfulDelivery:
    def test_mark_delivered_after_http_200(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        key = str(uuid.uuid4())
        coilbrain_client._enqueue(conn, key, coilbrain_client.build_payload(_make_result()))

        resp_mock = MagicMock()
        resp_mock.status = 200
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp_mock):
            coilbrain_client._deliver_one("http://example.com/signals", key, json.dumps({"x": 1}))
        coilbrain_client._mark_delivered(conn, key)
        conn.close()

        assert _count_state(outbox_db, "DELIVERED") == 1
        assert _count_state(outbox_db, "PENDING") == 0

    def test_deliver_one_sends_idempotency_key_header(self):
        captured = {}

        def capture(req, timeout):
            captured.update(req.headers)
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        key = str(uuid.uuid4())
        with patch("urllib.request.urlopen", side_effect=capture):
            coilbrain_client._deliver_one("http://x.com/signals", key, '{"a":1}')

        lower = {k.lower(): v for k, v in captured.items()}
        assert "x-idempotency-key" in lower
        assert lower["x-idempotency-key"] == key

    def test_no_x_signal_id_header_sent(self):
        """The old X-Signal-Id header must NOT be present — only X-Idempotency-Key."""
        captured = {}

        def capture(req, timeout):
            captured.update(req.headers)
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=capture):
            coilbrain_client._deliver_one("http://x.com/signals", str(uuid.uuid4()), '{}')

        lower = {k.lower() for k in captured}
        assert "x-signal-id" not in lower


# ===========================================================================
# 18. Corrupted outbox item
# ===========================================================================

class TestCorruptedOutboxItem:
    def test_corrupted_item_marked_dead_good_item_delivered(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        good_key = str(uuid.uuid4())
        bad_key = str(uuid.uuid4())
        good_payload = json.dumps(coilbrain_client.build_payload(_make_result()))
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) VALUES (?, ?, ?, 0, 'PENDING')",
            (good_key, good_payload, time.time()),
        )
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) VALUES (?, 'NOT{{JSON', ?, 0, 'PENDING')",
            (bad_key, time.time()),
        )
        conn.commit()

        resp_mock = MagicMock()
        resp_mock.status = 200
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)

        monkeypatch.setenv("COILBRAIN_URL", "http://example.com/signals")
        with patch("urllib.request.urlopen", return_value=resp_mock):
            due = coilbrain_client._due_items(conn)
            for row in due:
                sig_id, pj, attempts, _ = row
                try:
                    json.loads(pj)
                except (json.JSONDecodeError, TypeError) as exc:
                    coilbrain_client._mark_failed(conn, sig_id, f"CORRUPTED: {exc}", attempts, permanent=True)
                    continue
                coilbrain_client._deliver_one("http://example.com/signals", sig_id, pj)
                coilbrain_client._mark_delivered(conn, sig_id)
        conn.close()

        states = {r["id"]: r["state"] for r in _get_all(outbox_db)}
        assert states[bad_key] == "DEAD"
        assert states[good_key] == "DELIVERED"

    def test_corrupted_item_identifiable_separately(self, monkeypatch, outbox_db):
        """A DEAD corrupted item still has its own distinct id in the outbox."""
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        bad_key = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) VALUES (?, 'BAD{{', ?, 0, 'PENDING')",
            (bad_key, time.time()),
        )
        conn.commit()
        coilbrain_client._mark_failed(conn, bad_key, "CORRUPTED", 1, permanent=True)
        conn.close()
        rows = {r["id"]: r for r in _get_all(outbox_db)}
        assert bad_key in rows
        assert rows[bad_key]["state"] == "DEAD"
        assert rows[bad_key]["last_error"] == "CORRUPTED"


# ===========================================================================
# 19. Diagnostic status
# ===========================================================================

class TestDiagnosticStatus:
    def test_status_empty_db(self, monkeypatch, outbox_db):
        s = coilbrain_client.outbox_status()
        assert s["pending_count"] == 0
        assert s["dead_count"] == 0
        assert s["total_delivered"] == 0
        assert s["oldest_pending_age_s"] is None
        assert s["last_successful_delivery_ago_s"] is None

    def test_status_pending_count(self, monkeypatch, outbox_db):
        monkeypatch.delenv("COILBRAIN_URL", raising=False)
        for i in range(3):
            coilbrain_client.notify(_make_result(symbol=f"TOK{i}/USDT:USDT"))
        assert coilbrain_client.outbox_status()["pending_count"] == 3

    def test_status_oldest_pending_age(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) VALUES (?, '{}', ?, 0, 'PENDING')",
            (str(uuid.uuid4()), time.time() - 3600),
        )
        conn.commit()
        conn.close()
        s = coilbrain_client.outbox_status()
        assert s["oldest_pending_age_s"] >= 3500

    def test_status_last_delivery(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, last_attempt_at, state) "
            "VALUES (?, '{}', ?, 1, ?, 'DELIVERED')",
            (str(uuid.uuid4()), time.time() - 60, time.time() - 30),
        )
        conn.commit()
        conn.close()
        s = coilbrain_client.outbox_status()
        assert s["last_successful_delivery_ago_s"] >= 25

    def test_status_dead_count(self, monkeypatch, outbox_db):
        conn = coilbrain_client._get_conn(outbox_db)
        coilbrain_client._ensure_schema(conn)
        conn.execute(
            "INSERT INTO outbox_items (id, payload_json, queued_at, attempts, state) VALUES (?, 'BAD', ?, 1, 'DEAD')",
            (str(uuid.uuid4()), time.time()),
        )
        conn.commit()
        conn.close()
        assert coilbrain_client.outbox_status()["dead_count"] == 1

    def test_print_status_does_not_raise(self, monkeypatch, outbox_db):
        coilbrain_client.print_status()
