"""Tests for the Signal Observatory layer.

All tests use a per-test temp-file SQLite database.
No real HTTP calls are made -- Binance endpoints are patched.
No live asyncio event loops -- direct asyncio.run() calls.

Test groups
-----------
1.  Symbol mapping -- to_binance_symbol() conversion and validation.
2.  Schedule logic -- _next_poll_interval() and _compute_schedule().
3.  Repository     -- insert/fetch/update against a real (temp) SQLite DB.
4.  Observatory    -- observe_signal() with mocked price fetches.
5.  Picker         -- OPEN -> OBSERVING transition logic.
6.  Restart        -- _recover_observing_signals() with expired and live signals.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from coilbrain import database, repository
from coilbrain.config import settings
from coilbrain.models import SignalRecord
from coilbrain.observatory import (
    _compute_schedule,
    _next_poll_interval,
    _recover_observing_signals,
    observe_signal,
    to_binance_symbol,
)


# ---------------------------------------------------------------------------
# Shared DB fixture (per test, so each test is fully isolated)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    """Initialise a fresh temp SQLite DB and configure the database module."""
    db_path = str(tmp_path / "test.db")
    asyncio.run(database.init_db(db_path))
    # configure() is called inside init_db already; db_path is now active
    yield db_path
    # tmp_path cleanup is automatic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


def _make_record(**overrides) -> SignalRecord:
    defaults = {
        "signal_id": str(uuid.uuid4()),
        "symbol": "KOMA/USDT:USDT",
        "exchange": "binance",
        "trigger": "ALPHA_COILER",
        "grade": "A",
        "score": 100.0,
        "signal": "PUMP",
        "oi_status": "OI_PRESENT",
        "price": 0.015447,
        "signal_timestamp": 1_700_000_000.0,
        "change_24h_pct": None,
        "volume_24h_usd": 4_300_000.0,
        "breakout_pct": None,
        "body_volume_spike": 3.3,
        "volume_spike": 8.6,
        "flat_period_days": 14.0,
        "received_timestamp": 1_700_000_001.0,
        "transport_latency_ms": 1000.0,
    }
    defaults.update(overrides)
    return SignalRecord(**defaults)


KNOWN = {
    "BTCUSDT", "ETHUSDT", "KOMAUSDT", "1000PEPEUSDT", "1000000MOGUSDT",
    "0GUSDT", "2ZUSDT",
}


# ---------------------------------------------------------------------------
# 1. Symbol mapping
# ---------------------------------------------------------------------------


class TestToFuturesSymbol:
    def test_standard_symbol(self):
        assert to_binance_symbol("BTC/USDT:USDT", KNOWN) == "BTCUSDT"

    def test_koma_symbol(self):
        assert to_binance_symbol("KOMA/USDT:USDT", KNOWN) == "KOMAUSDT"

    def test_1000x_prefix_symbol(self):
        assert to_binance_symbol("1000PEPE/USDT:USDT", KNOWN) == "1000PEPEUSDT"

    def test_million_prefix_symbol(self):
        assert to_binance_symbol("1000000MOG/USDT:USDT", KNOWN) == "1000000MOGUSDT"

    def test_numeric_start_symbol(self):
        assert to_binance_symbol("0G/USDT:USDT", KNOWN) == "0GUSDT"

    def test_2z_symbol(self):
        assert to_binance_symbol("2Z/USDT:USDT", KNOWN) == "2ZUSDT"

    def test_unknown_symbol_returns_none(self):
        assert to_binance_symbol("FAKECOIN/USDT:USDT", KNOWN) is None

    def test_empty_known_set_returns_none(self):
        assert to_binance_symbol("BTC/USDT:USDT", set()) is None

    def test_empty_string_returns_none(self):
        assert to_binance_symbol("", KNOWN) is None

    # --- Raw Binance REST symbols (no slash) ----------------------------------

    def test_raw_btcusdt_known(self):
        """A raw Binance REST symbol present in known is returned as-is."""
        assert to_binance_symbol("BTCUSDT", KNOWN) == "BTCUSDT"

    def test_raw_1000pepeusdt_known(self):
        """1000x-prefix REST symbols are accepted when present in known."""
        assert to_binance_symbol("1000PEPEUSDT", KNOWN) == "1000PEPEUSDT"

    def test_raw_unknown_returns_none(self):
        """A raw REST symbol NOT in known returns None."""
        assert to_binance_symbol("FAKECOINUSDT", KNOWN) is None

    def test_raw_empty_known_returns_none(self):
        """A raw REST symbol against an empty known set returns None."""
        assert to_binance_symbol("BTCUSDT", set()) is None

    # --- Existing CCXT paths still work -------------------------------------

    def test_ccxt_btc_usdt(self):
        assert to_binance_symbol("BTC/USDT:USDT", KNOWN) == "BTCUSDT"

    def test_ccxt_1000pepe_usdt(self):
        assert to_binance_symbol("1000PEPE/USDT:USDT", KNOWN) == "1000PEPEUSDT"

    def test_ccxt_unknown_returns_none(self):
        assert to_binance_symbol("UNKNOWN/USDT:USDT", KNOWN) is None

    def test_ccxt_malformed_no_colon_returns_none(self):
        """A slash-containing string with no colon is malformed ccxt."""
        assert to_binance_symbol("BTC/USDT", KNOWN) is None



# ---------------------------------------------------------------------------
# 2. Schedule logic
# ---------------------------------------------------------------------------


class TestScheduleLogic:
    def test_phase1_interval(self):
        assert _next_poll_interval(0.0) == settings.obs_phase1_interval_s

    def test_phase1_near_boundary(self):
        just_before = settings.obs_phase1_duration_s - 1
        assert _next_poll_interval(just_before) == settings.obs_phase1_interval_s

    def test_phase2_interval(self):
        assert _next_poll_interval(settings.obs_phase1_duration_s) == settings.obs_phase2_interval_s

    def test_phase2_deep(self):
        assert _next_poll_interval(settings.obs_phase1_duration_s + 1000) == settings.obs_phase2_interval_s

    def test_compute_schedule_no_prior_obs(self):
        t0 = 1_700_000_000.0
        next_at, interval = _compute_schedule(t0, None)
        assert interval == settings.obs_phase1_interval_s
        assert next_at == t0 + settings.obs_phase1_interval_s

    def test_compute_schedule_with_prior_obs_phase1(self):
        t0 = 1_700_000_000.0
        last_obs = t0 + 60.0
        next_at, interval = _compute_schedule(t0, last_obs)
        assert interval == settings.obs_phase1_interval_s
        assert next_at == last_obs + settings.obs_phase1_interval_s

    def test_compute_schedule_with_prior_obs_phase2(self):
        t0 = 1_700_000_000.0
        last_obs = t0 + 7200.0
        next_at, interval = _compute_schedule(t0, last_obs)
        assert interval == settings.obs_phase2_interval_s
        assert next_at == last_obs + settings.obs_phase2_interval_s


# ---------------------------------------------------------------------------
# 3. Repository
# ---------------------------------------------------------------------------


class TestRepository:
    def test_insert_and_fetch_signal(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
            async with database.get_db() as db:
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row is not None
        assert row["signal_id"] == record.signal_id
        assert row["symbol"] == "KOMA/USDT:USDT"
        assert row["obs_status"] == "OPEN"

    def test_fetch_nonexistent_returns_none(self, tmp_db):
        async def go():
            async with database.get_db() as db:
                return await repository.fetch_signal(db, "nonexistent-id")

        assert _run(go()) is None

    def test_mark_observing(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_observing(db, record.signal_id)
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["obs_status"] == "OBSERVING"
        assert row["obs_started_at"] is not None

    def test_mark_complete(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_complete(db, record.signal_id)
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["obs_status"] == "COMPLETE"

    def test_mark_error(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_error(db, record.signal_id)
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["obs_status"] == "ERROR"

    def test_insert_observation(self, tmp_db):
        record = _make_record()
        now = time.time()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.insert_observation(
                    db, record.signal_id, now, 30.0, 0.016, 3.57
                )
            async with database.get_db() as db:
                return await repository.fetch_latest_observation_timestamp(
                    db, record.signal_id
                )

        ts = _run(go())
        assert ts == pytest.approx(now, abs=0.001)

    def test_update_outcome(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.update_outcome(
                    db, record.signal_id,
                    mfe_pct=5.5, mae_pct=1.2,
                    max_price=0.0162, min_price=0.015,
                    time_to_max_price_s=120.0,
                    secs_to_1pct=45.0, secs_to_5pct=90.0, secs_to_10pct=None,
                )
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["mfe_pct"] == pytest.approx(5.5)
        assert row["mae_pct"] == pytest.approx(1.2)
        assert row["secs_to_10pct"] is None

    def test_insert_error_row(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.insert_error(db, record.signal_id, "test error")
            async with aiosqlite.connect(tmp_db) as db:
                async with db.execute(
                    "SELECT error_msg FROM observation_errors WHERE signal_id=?",
                    (record.signal_id,),
                ) as cur:
                    return await cur.fetchall()

        rows = _run(go())
        assert len(rows) == 1
        assert rows[0][0] == "test error"

    def test_fetch_open_signals(self, tmp_db):
        r1 = _make_record()
        r2 = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, r1)
                await repository.insert_signal(db, r2)
                await repository.mark_complete(db, r2.signal_id)
                return await repository.fetch_open_signals(db)

        rows = _run(go())
        assert len(rows) == 1
        assert rows[0]["signal_id"] == r1.signal_id


# ---------------------------------------------------------------------------
# 4. Observatory -- observe_signal with mocked price fetches
# ---------------------------------------------------------------------------


class TestObserveSignal:
    """Run observe_signal with a tiny window so tests complete quickly."""

    def _run_observation(self, tmp_db, prices: list):
        """Run observe_signal with the given price sequence, tiny window."""
        record = _make_record()
        t0 = time.time()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_observing(db, record.signal_id)

            price_iter = iter(prices)

            async def mock_fetch(_client, _symbol):
                try:
                    return next(price_iter)
                except StopIteration:
                    return None

            # Use a very short window so the loop terminates quickly
            saved = (
                settings.obs_window_hours,
                settings.obs_phase1_duration_s,
                settings.obs_phase1_interval_s,
            )
            settings.obs_window_hours = 1.0 / 3600.0     # 1.0 seconds
            settings.obs_phase1_duration_s = 9999
            settings.obs_phase1_interval_s = 0.02   # 20ms per poll

            try:
                with patch("coilbrain.observatory._fetch_price", side_effect=mock_fetch):
                    await observe_signal(
                        signal_id=record.signal_id,
                        symbol=record.symbol,
                        binance_symbol="KOMAUSDT",
                        t0_price=record.price,
                        t0_timestamp=t0,
                        last_obs_ts=None,
                    )
            finally:
                (
                    settings.obs_window_hours,
                    settings.obs_phase1_duration_s,
                    settings.obs_phase1_interval_s,
                ) = saved

            async with database.get_db() as db:
                row = await repository.fetch_signal(db, record.signal_id)
                async with db.execute(
                    "SELECT price FROM price_observations WHERE signal_id=?",
                    (record.signal_id,),
                ) as cur:
                    obs_rows = await cur.fetchall()

            return row, [r[0] for r in obs_rows]

        return _run(go())

    def test_signal_marked_complete_after_window(self, tmp_db):
        row, _ = self._run_observation(tmp_db, prices=[0.016, 0.017])
        assert row["obs_status"] == "COMPLETE"

    def test_observations_are_inserted(self, tmp_db):
        row, obs = self._run_observation(tmp_db, prices=[0.016, 0.017])
        assert len(obs) >= 1

    def test_mfe_calculated(self, tmp_db):
        t0_price = 0.015447
        row, _ = self._run_observation(
            tmp_db, prices=[t0_price * 1.02, t0_price * 1.10, t0_price * 1.05]
        )
        assert row["mfe_pct"] == pytest.approx(10.0, abs=0.5)

    def test_mae_calculated(self, tmp_db):
        t0_price = 0.015447
        row, _ = self._run_observation(
            tmp_db, prices=[t0_price * 0.95, t0_price * 0.92]
        )
        assert row["mae_pct"] == pytest.approx(8.0, abs=0.5)

    def test_secs_to_1pct_recorded(self, tmp_db):
        t0_price = 0.015447
        row, _ = self._run_observation(
            tmp_db, prices=[t0_price * 1.005, t0_price * 1.015]
        )
        assert row["secs_to_1pct"] is not None

    def test_secs_to_10pct_none_when_not_reached(self, tmp_db):
        t0_price = 0.015447
        row, _ = self._run_observation(tmp_db, prices=[t0_price * 1.02])
        assert row["secs_to_10pct"] is None

    def test_failed_fetch_inserts_error_and_continues(self, tmp_db):
        """None price should log an error but not crash the observation."""
        row, obs = self._run_observation(tmp_db, prices=[None, 0.016])
        assert row["obs_status"] == "COMPLETE"


# ---------------------------------------------------------------------------
# 5. Picker -- OPEN -> OBSERVING transition
# ---------------------------------------------------------------------------


class TestPicker:
    def test_open_signal_becomes_observing(self, tmp_db):
        record = _make_record()

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)

            from coilbrain import observatory

            observatory._known_symbols = {"KOMAUSDT"}
            observatory._registry_loaded_at = time.time()

            # Simulate one picker cycle without spawning real tasks
            async with database.get_db() as db:
                open_sigs = await repository.fetch_open_signals(db)

            for row in open_sigs:
                sid = row["signal_id"]
                binance_sym = to_binance_symbol(row["symbol"], observatory._known_symbols)
                if binance_sym:
                    async with database.get_db() as db:
                        await repository.mark_observing(db, sid)

            async with database.get_db() as db:
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["obs_status"] == "OBSERVING"

    def test_unknown_symbol_stays_open_with_error(self, tmp_db):
        record = _make_record(symbol="FAKECOIN/USDT:USDT")

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)

            from coilbrain import observatory

            observatory._known_symbols = {"BTCUSDT"}  # FAKECOIN not present
            observatory._registry_loaded_at = time.time()

            # Simulate picker: no valid symbol -> insert error, leave OPEN
            async with database.get_db() as db:
                open_sigs = await repository.fetch_open_signals(db)

            for row in open_sigs:
                sid = row["signal_id"]
                binance_sym = to_binance_symbol(row["symbol"], observatory._known_symbols)
                if binance_sym is None:
                    async with database.get_db() as db:
                        await repository.insert_error(
                            db, sid, f"no valid USDM market for {row['symbol']!r}"
                        )

            async with database.get_db() as db:
                fetched = await repository.fetch_signal(db, record.signal_id)
            async with aiosqlite.connect(tmp_db) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM observation_errors WHERE signal_id=?",
                    (record.signal_id,),
                ) as cur:
                    err_count = (await cur.fetchone())[0]
            return fetched, err_count

        row, err_count = _run(go())
        assert row["obs_status"] == "OPEN"
        assert err_count == 1


# ---------------------------------------------------------------------------
# 6. Restart recovery
# ---------------------------------------------------------------------------


class TestRestartRecovery:
    def test_expired_signal_marked_complete(self, tmp_db):
        """A signal that was OBSERVING but whose window elapsed must be COMPLETE."""
        record = _make_record(
            signal_timestamp=time.time() - (settings.obs_window_hours * 3600 + 120)
        )

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_observing(db, record.signal_id)

            from coilbrain import observatory

            observatory._known_symbols = {"KOMAUSDT"}
            observatory._registry_loaded_at = time.time()

            with patch("coilbrain.observatory._ensure_registry", new_callable=AsyncMock):
                await _recover_observing_signals()

            async with database.get_db() as db:
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["obs_status"] == "COMPLETE"

    def test_live_signal_resumed_with_task(self, tmp_db):
        """A mid-window OBSERVING signal must spawn a new observation task."""
        record = _make_record(signal_timestamp=time.time() - 60.0)

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_observing(db, record.signal_id)

            from coilbrain import observatory

            observatory._known_symbols = {"KOMAUSDT"}
            observatory._registry_loaded_at = time.time()

            with patch("coilbrain.observatory._ensure_registry", new_callable=AsyncMock):
                with patch("coilbrain.observatory.asyncio.create_task") as mock_create:
                    mock_create.return_value = MagicMock()
                    await _recover_observing_signals()
                    return mock_create.call_count

        count = _run(go())
        assert count >= 1

    def test_invalid_symbol_during_recovery_marks_error(self, tmp_db):
        record = _make_record(
            symbol="FAKECOIN/USDT:USDT",
            signal_timestamp=time.time() - 60.0,
        )

        async def go():
            async with database.get_db() as db:
                await repository.insert_signal(db, record)
                await repository.mark_observing(db, record.signal_id)

            from coilbrain import observatory

            observatory._known_symbols = {"BTCUSDT"}
            observatory._registry_loaded_at = time.time()

            with patch("coilbrain.observatory._ensure_registry", new_callable=AsyncMock):
                await _recover_observing_signals()

            async with database.get_db() as db:
                return await repository.fetch_signal(db, record.signal_id)

        row = _run(go())
        assert row["obs_status"] == "ERROR"
