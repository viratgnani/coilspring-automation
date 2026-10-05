"""Tests for microstructure_recorder.py.

All external dependencies (fetcher, DB) are mocked.
Tests verify: lifecycle, symbol deduplication, task exit, failure isolation,
symbol cap, and backward-compat resolution of NULL binance_symbol.
"""

from __future__ import annotations

import asyncio
import uuid
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coilbrain import database
from coilbrain import microstructure_recorder as recorder
from coilbrain import repository
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.models import SignalRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_recorder_state():
    """Ensure module-level recorder state is clean for each test."""
    recorder._ms_picker_task = None
    recorder._active_symbol_tasks.clear()
    recorder._last_purge_ts = 0.0
    yield
    # Cleanup after test
    for task in list(recorder._active_symbol_tasks.values()):
        task.cancel()
    recorder._active_symbol_tasks.clear()
    if recorder._ms_picker_task and not recorder._ms_picker_task.done():
        recorder._ms_picker_task.cancel()
    recorder._ms_picker_task = None
    recorder._last_purge_ts = 0.0


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
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "trigger": "COILED_SPRING_V2",
        "grade": "A",
        "score": 100.0,
        "signal": "PUMP",
        "oi_status": "OI_PRESENT",
        "price": 65000.0,
        "signal_timestamp": time.time() - 10.0,
        "change_24h_pct": None,
        "volume_24h_usd": 1_000_000.0,
        "breakout_pct": None,
        "body_volume_spike": 3.0,
        "volume_spike": 8.0,
        "flat_period_days": 14.0,
        "received_timestamp": time.time(),
        "transport_latency_ms": 100.0,
    }
    defaults.update(overrides)
    return SignalRecord(**defaults)


def _make_snap(symbol: str = "BTCUSDT") -> MicrostructureSnapshot:
    return MicrostructureSnapshot(
        binance_symbol=symbol, snapshot_ts=time.time(),
        oi_usdt=1_000_000.0, oi_delta_pct=None,
        funding_rate=0.0001, mark_price=65000.0, next_funding_ts=time.time() + 28800,
        best_bid=64999.0, best_ask=65001.0,
        spread_abs=2.0, spread_pct=0.003,
        bid_depth_usdt=100_000.0, ask_depth_usdt=90_000.0, ob_imbalance=0.05,
        trade_count=None, traded_vol_usdt=None, buy_vol_usdt=None,
        sell_vol_usdt=None, trade_imbalance=None,
    )


async def _insert_observing(db_path: str, symbol: str = "BTCUSDT", binance_symbol: str = "BTCUSDT"):
    rec = _make_signal(symbol=symbol)
    async with database.get_db() as db:
        await repository.insert_signal(db, rec)
        await repository.mark_observing(db, rec.signal_id)
        await repository.set_binance_symbol(db, rec.signal_id, binance_symbol)
    return rec.signal_id


# ---------------------------------------------------------------------------
# 1. start() / stop() lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_creates_picker_task(self, tmp_db):
        async def _run():
            await recorder.start()
            assert recorder._ms_picker_task is not None
            assert not recorder._ms_picker_task.done()
            await recorder.stop()
            # Allow aiosqlite worker threads (spawned by the picker's first
            # DB cycle) to post their call_soon_threadsafe callbacks before
            # asyncio.run() closes the event loop.
            await asyncio.sleep(0.3)
        asyncio.run(_run())

    def test_stop_cancels_picker_task(self, tmp_db):
        async def _run():
            await recorder.start()
            task = recorder._ms_picker_task
            await recorder.stop()
            await asyncio.sleep(0.3)  # drain aiosqlite threads from picker cycle
            assert task.done()
        asyncio.run(_run())

    def test_stop_clears_active_tasks(self, tmp_db):
        async def _run():
            await recorder.start()
            # Inject a fake symbol task
            fake = asyncio.create_task(asyncio.sleep(100), name="ms-BTCUSDT")
            recorder._active_symbol_tasks["BTCUSDT"] = fake
            await recorder.stop()
            await asyncio.sleep(0.3)  # drain aiosqlite threads from picker cycle
            assert len(recorder._active_symbol_tasks) == 0
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. _symbol_loop exits when no active signals remain
# ---------------------------------------------------------------------------

class TestSymbolLoopExit:
    def test_loop_exits_when_no_signals(self):
        """_symbol_loop should exit promptly if no OBSERVING signals exist.

        The database layer is mocked so that no real aiosqlite connections are
        created.  This avoids a race between the aiosqlite worker thread and
        asyncio.run() closing the event loop, which produced a
        PytestUnhandledThreadExceptionWarning.  The logic under test lives
        entirely in the recorder (break on empty active list), not in the DB.
        """
        # Fake async context manager for database.get_db()
        _mock_ctx = MagicMock()
        _mock_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
        _mock_ctx.__aexit__ = AsyncMock(return_value=False)

        async def _run():
            with patch(
                "coilbrain.microstructure_recorder.fetch_all",
                new=AsyncMock(return_value=_make_snap()),
            ):
                with patch("coilbrain.database.get_db", return_value=_mock_ctx):
                    with patch(
                        "coilbrain.microstructure_repository"
                        ".fetch_active_signals_for_symbol",
                        new=AsyncMock(return_value=[]),
                    ):
                        task = asyncio.create_task(
                            recorder._symbol_loop("BTCUSDT"), name="ms-BTCUSDT"
                        )
                        recorder._active_symbol_tasks["BTCUSDT"] = task
                        # Task must exit quickly because active list is empty.
                        await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
                        assert task.done()
        asyncio.run(_run())


    def test_loop_fetches_snapshot_while_signal_active(self, tmp_db):
        """Loop should call fetch_all at least once while signal is OBSERVING."""
        sid = _run(_insert_observing(tmp_db))

        fetch_calls = []

        async def mock_fetch_all(client, symbol, since_ts, prev_oi_usdt=None):
            fetch_calls.append(symbol)
            # After first fetch, mark signal complete so loop exits
            async with database.get_db() as db:
                await repository.mark_complete(db, sid)
            return _make_snap(symbol)

        async def _run2():
            with patch("coilbrain.microstructure_recorder.fetch_all", side_effect=mock_fetch_all):
                with patch("coilbrain.microstructure_recorder.settings") as ms:
                    ms.ms_poll_interval_s = 0.01
                    ms.ms_stale_threshold_multiplier = 3.0
                    task = asyncio.create_task(recorder._symbol_loop("BTCUSDT"))
                    recorder._active_symbol_tasks["BTCUSDT"] = task
                    await asyncio.wait_for(task, timeout=2.0)
        asyncio.run(_run2())
        assert "BTCUSDT" in fetch_calls


# ---------------------------------------------------------------------------
# 3. No duplicate tasks per symbol
# ---------------------------------------------------------------------------

class TestNoduplicateTasks:
    def test_picker_does_not_spawn_duplicate_task(self, tmp_db):
        """If a symbol task already exists, the picker must not spawn another."""
        sid = _run(_insert_observing(tmp_db))

        # Pre-register a fake task for BTCUSDT
        fake_task = MagicMock()
        fake_task.done.return_value = False
        recorder._active_symbol_tasks["BTCUSDT"] = fake_task

        spawn_count = [0]
        original_create_task = asyncio.create_task

        def mock_create_task(coro, **kwargs):
            if "ms-BTCUSDT" in kwargs.get("name", ""):
                spawn_count[0] += 1
            return original_create_task(coro, **kwargs)

        async def _run2():
            with patch("coilbrain.microstructure_recorder.settings") as ms:
                ms.ms_picker_interval_s = 9999.0
                ms.ms_max_symbols = 20
                with patch("asyncio.create_task", side_effect=mock_create_task):
                    # Run one picker iteration manually
                    try:
                        await asyncio.wait_for(recorder._ms_picker_loop(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
        asyncio.run(_run2())
        assert spawn_count[0] == 0, "Picker spawned a duplicate task"


# ---------------------------------------------------------------------------
# 4. Symbol cap enforced
# ---------------------------------------------------------------------------

class TestSymbolCap:
    def test_cap_limits_spawned_tasks_and_cancels_old_ones(self, tmp_db):
        """With ms_max_symbols=2, tests that it spawns only 2 and cancels old ones when pushed out."""
        async def _setup():
            for i, sym in enumerate(["SYM1", "SYM2", "SYM3"]):
                rec = _make_signal(symbol=sym)
                # Stagger the timestamps so SYM3 is newest, SYM1 is oldest
                rec.signal_timestamp = time.time() + i * 10
                async with database.get_db() as db:
                    await repository.insert_signal(db, rec)
                    await repository.mark_observing(db, rec.signal_id)
                    await repository.set_binance_symbol(db, rec.signal_id, sym)
        asyncio.run(_setup())

        spawned = []
        cancelled = []

        async def _run2():
            orig_interval = recorder.settings.ms_picker_interval_s
            orig_max = recorder.settings.ms_max_symbols
            recorder.settings.ms_picker_interval_s = 9999.0
            recorder.settings.ms_max_symbols = 2
            try:
                original_create_task = asyncio.create_task
                
                def mock_create_task(coro, **kwargs):
                    name = kwargs.get("name", "")
                    if name.startswith("ms-"):
                        spawned.append(name)
                        print(f"Spawned {name}")
                    task = original_create_task(coro, **kwargs)
                    
                    original_cancel = task.cancel
                    def mock_cancel(*cargs, **ckwargs):
                        print(f"Cancelled {name}")
                        cancelled.append(name)
                        return original_cancel(*cargs, **ckwargs)
                    task.cancel = mock_cancel
                    
                    return task
                
                with patch("asyncio.create_task", side_effect=mock_create_task):
                    # First run: SYM2 and SYM3 should be selected (they are the newest 2)
                    try:
                        await asyncio.wait_for(recorder._ms_picker_loop(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                        
                    # Now let's add SYM4 which is even newer
                    rec = _make_signal(symbol="SYM4")
                    rec.signal_timestamp = time.time() + 100
                    async with database.get_db() as db:
                        await repository.insert_signal(db, rec)
                        await repository.mark_observing(db, rec.signal_id)
                        await repository.set_binance_symbol(db, rec.signal_id, "SYM4")
                        
                    # Second run: SYM4 and SYM3 should be selected. SYM2 should be cancelled.
                    try:
                        await asyncio.wait_for(recorder._ms_picker_loop(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
            finally:
                recorder.settings.ms_picker_interval_s = orig_interval
                recorder.settings.ms_max_symbols = orig_max

        asyncio.run(_run2())

        assert "ms-SYM3" in spawned
        assert "ms-SYM2" in spawned
        assert "ms-SYM1" not in spawned  # SYM1 was oldest, never spawned
        
        assert "ms-SYM4" in spawned      # SYM4 was added and spawned
        
        # Since SYM4 and SYM3 are the newest 2, SYM2 should have been cancelled
        assert "ms-SYM2" in cancelled


# ---------------------------------------------------------------------------
# 5. Fetch failure does not stop the loop
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    def test_fetch_error_does_not_kill_loop(self, tmp_db):
        """A fetch_all exception should be caught; loop continues."""
        sid = _run(_insert_observing(tmp_db))

        call_count = [0]

        async def failing_then_completing(client, symbol, since_ts, prev_oi_usdt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated network failure")
            # Second call succeeds but marks signal complete so loop exits
            async with database.get_db() as db:
                await repository.mark_complete(db, sid)
            return _make_snap(symbol)

        async def _run2():
            with patch("coilbrain.microstructure_recorder.fetch_all", side_effect=failing_then_completing):
                with patch("coilbrain.microstructure_recorder.settings") as ms:
                    ms.ms_poll_interval_s = 0.01
                    ms.ms_stale_threshold_multiplier = 3.0
                    task = asyncio.create_task(recorder._symbol_loop("BTCUSDT"))
                    recorder._active_symbol_tasks["BTCUSDT"] = task
                    await asyncio.wait_for(task, timeout=2.0)
        asyncio.run(_run2())
        assert call_count[0] >= 2

    def test_recorder_crash_does_not_affect_observatory_tables(self, tmp_db):
        """A recorder failure must NOT write to observation_errors."""
        async def _run2():
            with patch(
                "coilbrain.microstructure_recorder.fetch_all",
                side_effect=RuntimeError("crash"),
            ):
                with patch("coilbrain.microstructure_recorder.settings") as ms:
                    ms.ms_poll_interval_s = 0.01
                    ms.ms_stale_threshold_multiplier = 1.0
                    task = asyncio.create_task(recorder._symbol_loop("BTCUSDT"))
                    recorder._active_symbol_tasks["BTCUSDT"] = task
                    await asyncio.sleep(0.1)
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            async with database.get_db() as db:
                async with db.execute("SELECT count(*) FROM observation_errors") as cur:
                    return (await cur.fetchone())[0]
        obs_errors = asyncio.run(_run2())
        assert obs_errors == 0, "Recorder failure must not write to observation_errors"


# ---------------------------------------------------------------------------
# 6. MYX Regression Test
# ---------------------------------------------------------------------------

class TestMYXRegression:
    def test_myx_gets_coverage_when_in_top_20(self, tmp_db):
        """
        Reproduce the MYX failure mode: if MYX has an active OBSERVING signal and
        capacity allows (within top 20), it must be selected for coverage and its
        task spawned. It should not be silently ignored.
        """
        async def _setup():
            # Insert 20 filler symbols
            for i in range(20):
                sym = f"FILLER{i}USDT"
                rec = _make_signal(symbol=sym)
                rec.signal_timestamp = time.time() - 100 + i
                async with database.get_db() as db:
                    await repository.insert_signal(db, rec)
                    await repository.mark_observing(db, rec.signal_id)
                    await repository.set_binance_symbol(db, rec.signal_id, sym)
                    
            # Insert MYX as the absolute newest
            rec = _make_signal(symbol="MYXUSDT")
            rec.signal_timestamp = time.time()
            async with database.get_db() as db:
                await repository.insert_signal(db, rec)
                await repository.mark_observing(db, rec.signal_id)
                await repository.set_binance_symbol(db, rec.signal_id, "MYXUSDT")
        asyncio.run(_setup())
        
        spawned = []
        
        async def _run2():
            orig_interval = recorder.settings.ms_picker_interval_s
            orig_max = recorder.settings.ms_max_symbols
            recorder.settings.ms_picker_interval_s = 9999.0
            recorder.settings.ms_max_symbols = 20
            try:
                original_create_task = asyncio.create_task
                def mock_create_task(coro, **kwargs):
                    name = kwargs.get("name", "")
                    if name.startswith("ms-"):
                        spawned.append(name)
                    return original_create_task(coro, **kwargs)
                
                with patch("asyncio.create_task", side_effect=mock_create_task):
                    try:
                        await asyncio.wait_for(recorder._ms_picker_loop(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
            finally:
                recorder.settings.ms_picker_interval_s = orig_interval
                recorder.settings.ms_max_symbols = orig_max
                
        asyncio.run(_run2())
        
        # MYX must be spawned since it is the newest and within the cap of 20
        assert "ms-MYXUSDT" in spawned
        # The oldest filler (FILLER0USDT) should have been pushed out
        assert "ms-FILLER0USDT" not in spawned


# ---------------------------------------------------------------------------
# 7. Lifecycle Churn Test
# ---------------------------------------------------------------------------

class TestLifecycleChurn:
    def test_picker_churn_maintains_cap_and_cleans_tasks(self, tmp_db):
        """
        Run the picker through several selection changes with >20 symbols,
        including repeated eviction and re-entry.
        Assert that _active_symbol_tasks never exceeds ms_max_symbols,
        cancelled tasks terminate, and a previously evicted symbol can be spawned again.
        """
        sequence_of_symbols = [
            # Cycle 1: 5 symbols (A, B, C, D, E). Cap is 3.
            # Expected running: A, B, C
            ["A", "B", "C", "D", "E"],
            
            # Cycle 2: F, A, B, C, D
            # Expected running: F, A, B. (C is evicted, F is spawned)
            ["F", "A", "B", "C", "D"],
            
            # Cycle 3: C, F, A, B, D
            # Expected running: C, F, A. (B is evicted, C re-enters!)
            ["C", "F", "A", "B", "D"],
            
            # Cycle 4: G, H, I, J
            # Expected running: G, H, I. (C, F, A all evicted)
            ["G", "H", "I", "J"],
        ]
        
        call_count = [0]
        c_task_cycle1 = [None]
        c_task_cycle3 = [None]
        
        async def mock_fetch_active_binance_symbols(db):
            idx = call_count[0]
            if idx > 0:
                expected_running = sequence_of_symbols[idx - 1][:3]
                actual_running = {sym for sym, task in recorder._active_symbol_tasks.items() if not task.done()}
                assert actual_running == set(expected_running), f"Cycle {idx}: {actual_running} != {set(expected_running)}"
                
                # Assert tasks are not done
                for sym in actual_running:
                    assert not recorder._active_symbol_tasks[sym].done()
                    
                if idx == 1:
                    c_task_cycle1[0] = recorder._active_symbol_tasks["C"]
                elif idx == 3:
                    c_task_cycle3[0] = recorder._active_symbol_tasks["C"]
            
            if idx >= len(sequence_of_symbols):
                # We are done. Raise CancelledError to gracefully exit the picker loop.
                raise asyncio.CancelledError()
                
            ret = sequence_of_symbols[idx]
            call_count[0] += 1
            return ret

        async def mock_fetch_observing_without_binance_symbol(db):
            return []
            
        async def mock_fetch_active_signals_for_symbol(db, sym):
            # Always return something so the _symbol_loop doesn't exit naturally
            return [{"signal_id": "fake", "t0_timestamp": 0}]

        # Mock fetch_all to just sleep so it doesn't hit Binance
        async def mock_fetch_all(*args, **kwargs):
            await asyncio.sleep(0.01)
            # return a dummy snap
            from coilbrain.microstructure_fetcher import MicrostructureSnapshot
            return MicrostructureSnapshot(
                binance_symbol="FAKE",
                snapshot_ts=time.time(),
                oi_usdt=0.0,
                oi_delta_pct=0.0,
                funding_rate=0.0,
                mark_price=1.0,
                next_funding_ts=0.0,
                best_bid=1.0,
                best_ask=1.0,
                spread_abs=0.0,
                spread_pct=0.0,
                bid_depth_usdt=0.0,
                ask_depth_usdt=0.0,
                ob_imbalance=0.0,
                trade_count=0,
                traded_vol_usdt=0.0,
                buy_vol_usdt=0.0,
                sell_vol_usdt=0.0,
                trade_imbalance=0.0
            )
            
        # We also need to mock insert_snapshot and insert_signal_snapshot so it doesn't fail on DB constraints
        async def mock_insert_snapshot(*args, **kwargs):
            return 1
            
        async def mock_insert_signal_snapshot(*args, **kwargs):
            pass

        async def _run():
            orig_interval = recorder.settings.ms_picker_interval_s
            orig_poll = recorder.settings.ms_poll_interval_s
            orig_max = recorder.settings.ms_max_symbols
            
            recorder.settings.ms_picker_interval_s = 0.05
            recorder.settings.ms_poll_interval_s = 0.05
            recorder.settings.ms_max_symbols = 3
            
            # Clear any leftover tasks
            recorder._active_symbol_tasks.clear()
            
            try:
                with patch("coilbrain.microstructure_recorder.micro_repo.fetch_active_binance_symbols", side_effect=mock_fetch_active_binance_symbols):
                    with patch("coilbrain.microstructure_recorder.micro_repo.fetch_observing_without_binance_symbol", side_effect=mock_fetch_observing_without_binance_symbol):
                        with patch("coilbrain.microstructure_recorder.micro_repo.fetch_active_signals_for_symbol", side_effect=mock_fetch_active_signals_for_symbol):
                            with patch("coilbrain.microstructure_recorder.fetch_all", side_effect=mock_fetch_all):
                                with patch("coilbrain.microstructure_recorder.micro_repo.insert_snapshot", side_effect=mock_insert_snapshot):
                                    with patch("coilbrain.microstructure_recorder.micro_repo.insert_signal_snapshot", side_effect=mock_insert_signal_snapshot):
                                        try:
                                            # Allow plenty of time for 4 cycles of 0.05s
                                            await asyncio.wait_for(recorder._ms_picker_loop(), timeout=10.0)
                                        except asyncio.CancelledError:
                                            pass
                                            
                                        # Now cleanup whatever tasks are left
                                        for task in recorder._active_symbol_tasks.values():
                                            task.cancel()
                                        
                                        if recorder._active_symbol_tasks:
                                            await asyncio.gather(*recorder._active_symbol_tasks.values(), return_exceptions=True)
                                        assert len(recorder._active_symbol_tasks) == 0

            finally:
                recorder.settings.ms_picker_interval_s = orig_interval
                recorder.settings.ms_poll_interval_s = orig_poll
                recorder.settings.ms_max_symbols = orig_max
                
        asyncio.run(_run())
        
        # Verify the entire sequence ran
        assert call_count[0] == len(sequence_of_symbols)
        
        # Verify that C was spawned, evicted, and spawned again as a new task
        assert c_task_cycle1[0] is not None
        assert c_task_cycle3[0] is not None
        assert c_task_cycle1[0] is not c_task_cycle3[0], "Re-entered symbol should have a new task instance"
        assert c_task_cycle1[0].done(), "Old evicted task should be terminated"
