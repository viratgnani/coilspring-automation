"""Tests for microstructure_repository.py.

All tests use a per-test temp-file SQLite DB (same fixture pattern as
test_observatory.py).  No real HTTP calls are made.
"""

from __future__ import annotations

import asyncio
import uuid
import time

import pytest

from coilbrain import database
from coilbrain import microstructure_repository as micro_repo
from coilbrain import repository
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.models import SignalRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    asyncio.run(database.init_db(db_path))
    yield db_path


def _run(coro):
    return asyncio.run(coro)


def _make_signal(**overrides) -> SignalRecord:
    defaults = {
        "signal_id": str(uuid.uuid4()),
        "symbol": "KOMA/USDT:USDT",
        "exchange": "binance",
        "trigger": "COILED_SPRING_V2",
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


def _make_snapshot(symbol: str = "BTCUSDT", **overrides) -> MicrostructureSnapshot:
    defaults = dict(
        binance_symbol=symbol,
        snapshot_ts=1_700_000_100.0,
        oi_usdt=6_500_000.0,
        oi_delta_pct=None,
        funding_rate=0.0001,
        mark_price=65000.0,
        next_funding_ts=1_700_008_000.0,
        best_bid=64999.0,
        best_ask=65001.0,
        spread_abs=2.0,
        spread_pct=0.003,
        bid_depth_usdt=100_000.0,
        ask_depth_usdt=90_000.0,
        ob_imbalance=0.053,
        trade_count=None,
        traded_vol_usdt=None,
        buy_vol_usdt=None,
        sell_vol_usdt=None,
        trade_imbalance=None,
    )
    defaults.update(overrides)
    return MicrostructureSnapshot(**defaults)


async def _insert_observing_signal(db_path: str, signal_id: str, symbol: str, binance_symbol: str) -> None:
    rec = _make_signal(signal_id=signal_id, symbol=symbol)
    async with database.get_db() as db:
        await repository.insert_signal(db, rec)
        await repository.mark_observing(db, signal_id)
        await repository.set_binance_symbol(db, signal_id, binance_symbol)


# ---------------------------------------------------------------------------
# 1. insert_snapshot / fetch_latest_snapshot_ts
# ---------------------------------------------------------------------------

class TestInsertSnapshot:
    def test_returns_integer_id(self, tmp_db):
        snap = _make_snapshot()
        async def _run():
            async with database.get_db() as db:
                return await micro_repo.insert_snapshot(db, snap)
        sid = asyncio.run(_run())
        assert isinstance(sid, int)
        assert sid >= 1

    def test_sequential_ids_increase(self, tmp_db):
        snap = _make_snapshot()
        async def _run():
            async with database.get_db() as db:
                id1 = await micro_repo.insert_snapshot(db, snap)
                id2 = await micro_repo.insert_snapshot(db, snap)
                return id1, id2
        id1, id2 = asyncio.run(_run())
        assert id2 > id1

    def test_null_fields_stored_as_null(self, tmp_db):
        snap = _make_snapshot(oi_usdt=None, funding_rate=None, best_bid=None)
        async def _run():
            async with database.get_db() as db:
                sid = await micro_repo.insert_snapshot(db, snap)
                async with db.execute(
                    "SELECT oi_usdt, funding_rate, best_bid FROM microstructure_snapshots WHERE id=?",
                    (sid,)
                ) as cur:
                    return dict(await cur.fetchone())
        row = asyncio.run(_run())
        assert row["oi_usdt"] is None
        assert row["funding_rate"] is None
        assert row["best_bid"] is None

    def test_fetch_latest_snapshot_ts(self, tmp_db):
        snap1 = _make_snapshot(snapshot_ts=1_000.0)
        snap2 = _make_snapshot(snapshot_ts=2_000.0)
        async def _run():
            async with database.get_db() as db:
                await micro_repo.insert_snapshot(db, snap1)
                await micro_repo.insert_snapshot(db, snap2)
                return await micro_repo.fetch_latest_snapshot_ts(db, "BTCUSDT")
        ts = asyncio.run(_run())
        assert ts == pytest.approx(2_000.0)

    def test_fetch_latest_ts_unknown_symbol_returns_none(self, tmp_db):
        async def _run():
            async with database.get_db() as db:
                return await micro_repo.fetch_latest_snapshot_ts(db, "UNKNOWNUSDT")
        assert asyncio.run(_run()) is None


# ---------------------------------------------------------------------------
# 2. insert_signal_snapshot / fetch_snapshots_for_signal
# ---------------------------------------------------------------------------

class TestSignalSnapshots:
    def test_junction_row_inserted(self, tmp_db):
        signal_id = str(uuid.uuid4())
        _run(_insert_observing_signal(tmp_db, signal_id, "KOMA/USDT:USDT", "KOMAUSDT"))

        async def _run2():
            snap = _make_snapshot("KOMAUSDT")
            async with database.get_db() as db:
                snap_id = await micro_repo.insert_snapshot(db, snap)
                await micro_repo.insert_signal_snapshot(db, signal_id, snap_id, elapsed_s=30.0)
                rows = await micro_repo.fetch_snapshots_for_signal(db, signal_id)
            return rows

        rows = asyncio.run(_run2())
        assert len(rows) == 1
        assert rows[0]["binance_symbol"] == "KOMAUSDT"
        assert rows[0]["elapsed_s"] == pytest.approx(30.0)

    def test_multiple_signals_share_one_snapshot(self, tmp_db):
        """Two signals for the same symbol should link to the same snapshot id."""
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())
        _run(_insert_observing_signal(tmp_db, sid1, "BTCUSDT", "BTCUSDT"))
        _run(_insert_observing_signal(tmp_db, sid2, "BTCUSDT", "BTCUSDT"))

        async def _run2():
            snap = _make_snapshot("BTCUSDT")
            async with database.get_db() as db:
                snap_id = await micro_repo.insert_snapshot(db, snap)
                await micro_repo.insert_signal_snapshot(db, sid1, snap_id, 30.0)
                await micro_repo.insert_signal_snapshot(db, sid2, snap_id, 35.0)
                r1 = await micro_repo.fetch_snapshots_for_signal(db, sid1)
                r2 = await micro_repo.fetch_snapshots_for_signal(db, sid2)
            return r1, r2

        r1, r2 = asyncio.run(_run2())
        assert len(r1) == 1
        assert len(r2) == 1
        # Both reference the same snapshot row
        assert r1[0]["id"] == r2[0]["id"]


# ---------------------------------------------------------------------------
# 3. fetch_active_binance_symbols / fetch_active_signals_for_symbol
# ---------------------------------------------------------------------------

class TestActiveSymbolQueries:
    def test_observing_signal_appears_in_active_symbols(self, tmp_db):
        signal_id = str(uuid.uuid4())
        _run(_insert_observing_signal(tmp_db, signal_id, "BTCUSDT", "BTCUSDT"))

        async def _run2():
            async with database.get_db() as db:
                return await micro_repo.fetch_active_binance_symbols(db)
        syms = asyncio.run(_run2())
        assert "BTCUSDT" in syms

    def test_signal_without_binance_symbol_not_in_active_symbols(self, tmp_db):
        """A signal marked OBSERVING but with NULL binance_symbol is excluded."""
        rec = _make_signal(signal_id=str(uuid.uuid4()), symbol="ETHUSDT")
        async def _setup():
            async with database.get_db() as db:
                await repository.insert_signal(db, rec)
                await repository.mark_observing(db, rec.signal_id)
                # deliberately NOT calling set_binance_symbol
        asyncio.run(_setup())

        async def _query():
            async with database.get_db() as db:
                return await micro_repo.fetch_active_binance_symbols(db)
        syms = asyncio.run(_query())
        assert "ETHUSDT" not in syms

    def test_fetch_active_signals_for_symbol_returns_correct_rows(self, tmp_db):
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())
        _run(_insert_observing_signal(tmp_db, sid1, "BTCUSDT", "BTCUSDT"))
        _run(_insert_observing_signal(tmp_db, sid2, "ETHUSDT", "ETHUSDT"))

        async def _run2():
            async with database.get_db() as db:
                return await micro_repo.fetch_active_signals_for_symbol(db, "BTCUSDT")
        rows = asyncio.run(_run2())
        assert len(rows) == 1
        assert rows[0]["signal_id"] == sid1

    def test_two_signals_same_symbol_both_returned(self, tmp_db):
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())
        _run(_insert_observing_signal(tmp_db, sid1, "BTCUSDT", "BTCUSDT"))
        _run(_insert_observing_signal(tmp_db, sid2, "BTCUSDT", "BTCUSDT"))

        async def _run2():
            async with database.get_db() as db:
                return await micro_repo.fetch_active_signals_for_symbol(db, "BTCUSDT")
        rows = asyncio.run(_run2())
        ids = {r["signal_id"] for r in rows}
        assert ids == {sid1, sid2}

    def test_fetch_observing_without_binance_symbol(self, tmp_db):
        rec = _make_signal(signal_id=str(uuid.uuid4()))
        async def _setup():
            async with database.get_db() as db:
                await repository.insert_signal(db, rec)
                await repository.mark_observing(db, rec.signal_id)
        asyncio.run(_setup())

        async def _query():
            async with database.get_db() as db:
                return await micro_repo.fetch_observing_without_binance_symbol(db)
        rows = asyncio.run(_query())
        sids = {r["signal_id"] for r in rows}
        assert rec.signal_id in sids


# ---------------------------------------------------------------------------
# 4. insert_micro_error
# ---------------------------------------------------------------------------

class TestMicroError:
    def test_error_inserted(self, tmp_db):
        async def _run2():
            async with database.get_db() as db:
                await micro_repo.insert_micro_error(db, "BTCUSDT", "oi", "test error")
                async with db.execute(
                    "SELECT * FROM microstructure_errors WHERE binance_symbol='BTCUSDT'"
                ) as cur:
                    return await cur.fetchall()
        rows = asyncio.run(_run2())
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["data_type"] == "oi"
        assert row["error_msg"] == "test error"

    def test_multiple_errors_for_same_symbol(self, tmp_db):
        async def _run2():
            async with database.get_db() as db:
                await micro_repo.insert_micro_error(db, "BTCUSDT", "oi", "err1")
                await micro_repo.insert_micro_error(db, "BTCUSDT", "book", "err2")
                async with db.execute(
                    "SELECT count(*) FROM microstructure_errors WHERE binance_symbol='BTCUSDT'"
                ) as cur:
                    return (await cur.fetchone())[0]
        count = asyncio.run(_run2())
        assert count == 2


# ---------------------------------------------------------------------------
# 5. purge_old_snapshots
# ---------------------------------------------------------------------------

class TestPurgeOldSnapshots:
    """Verify B2 fix: old snapshots and their junction rows are deleted;
    recent ones survive."""

    def _make_snap_at(self, ts: float, symbol: str = "BTCUSDT") -> MicrostructureSnapshot:
        return _make_snapshot(symbol, snapshot_ts=ts)

    def test_old_snapshots_are_deleted(self, tmp_db):
        """Snapshots older than the cutoff must be removed."""
        cutoff = 1_700_000_000.0
        old_ts  = cutoff - 1.0    # before cutoff → purged
        new_ts  = cutoff + 1.0    # after  cutoff → kept

        async def _run():
            async with database.get_db() as db:
                await micro_repo.insert_snapshot(db, self._make_snap_at(old_ts))
                await micro_repo.insert_snapshot(db, self._make_snap_at(new_ts))
                deleted = await micro_repo.purge_old_snapshots(db, older_than_ts=cutoff)
                async with db.execute(
                    "SELECT count(*) FROM microstructure_snapshots"
                ) as cur:
                    remaining = (await cur.fetchone())[0]
            return deleted, remaining

        deleted, remaining = asyncio.run(_run())
        assert deleted == 1
        assert remaining == 1

    def test_junction_rows_deleted_with_snapshot(self, tmp_db):
        """Junction rows for purged snapshots must also be removed."""
        signal_id = str(uuid.uuid4())
        _run(_insert_observing_signal(tmp_db, signal_id, "BTCUSDT", "BTCUSDT"))

        cutoff = 1_700_000_000.0

        async def _run2():
            async with database.get_db() as db:
                # Insert an old snapshot and link it to the signal
                snap_id = await micro_repo.insert_snapshot(
                    db, self._make_snap_at(cutoff - 1.0)
                )
                await micro_repo.insert_signal_snapshot(db, signal_id, snap_id, 30.0)
                # Purge
                await micro_repo.purge_old_snapshots(db, older_than_ts=cutoff)
                # Check junction table
                async with db.execute(
                    "SELECT count(*) FROM microstructure_signal_snapshots"
                ) as cur:
                    return (await cur.fetchone())[0]

        junction_count = asyncio.run(_run2())
        assert junction_count == 0, "Junction rows must be deleted with their snapshot"

    def test_recent_snapshots_survive_purge(self, tmp_db):
        """Snapshots at or after the cutoff are not touched."""
        cutoff = 1_700_000_000.0

        async def _run():
            async with database.get_db() as db:
                await micro_repo.insert_snapshot(db, self._make_snap_at(cutoff))
                await micro_repo.insert_snapshot(db, self._make_snap_at(cutoff + 3600))
                deleted = await micro_repo.purge_old_snapshots(db, older_than_ts=cutoff)
                async with db.execute(
                    "SELECT count(*) FROM microstructure_snapshots"
                ) as cur:
                    remaining = (await cur.fetchone())[0]
            return deleted, remaining

        deleted, remaining = asyncio.run(_run())
        assert deleted == 0
        assert remaining == 2

    def test_purge_empty_table_returns_zero(self, tmp_db):
        """Purging an empty table must return 0 without error."""
        async def _run():
            async with database.get_db() as db:
                return await micro_repo.purge_old_snapshots(db, older_than_ts=9_999_999_999.0)
        assert asyncio.run(_run()) == 0

    def test_purge_multiple_old_rows_returns_correct_count(self, tmp_db):
        """Count reflects the number of snapshot rows deleted, not junction rows."""
        cutoff = 1_700_000_000.0

        async def _run():
            async with database.get_db() as db:
                for offset in [-100, -200, -300]:
                    await micro_repo.insert_snapshot(
                        db, self._make_snap_at(cutoff + offset)
                    )
                return await micro_repo.purge_old_snapshots(db, older_than_ts=cutoff)

        assert asyncio.run(_run()) == 3
