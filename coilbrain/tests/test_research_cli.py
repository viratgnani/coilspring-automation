"""Tests for the Research CLI integration."""

import pytest
from unittest.mock import patch, AsyncMock
import argparse

from coilbrain.research.cli import _run, main

@pytest.fixture
def mock_regime_core():
    with patch("coilbrain.research.analysis.regime_core.build_regime_dataset", new_callable=AsyncMock) as m:
        # Mock what build_regime_dataset returns
        m.return_value = {
            "states": [],
            "dataset_summary": {"TOTAL_SIGNALS": 0}
        }
        yield m

@pytest.fixture
def mock_exporter():
    with patch("coilbrain.research.analysis.exporter.export_regime_results") as m:
        yield m
        
@pytest.fixture
def mock_print_case_study():
    with patch("coilbrain.research.cli._print_regime_case_study") as m:
        yield m


import asyncio

def test_regime_cli_dispatch(mock_regime_core, mock_exporter):
    async def _do():
        args = argparse.Namespace(
            generate=False,
            symbol=None,
            signal_id=None,
            summary=False,
            runner_threshold=2.0,
            regime=True,
            regime_symbol=None,
            regime_signal_id=None,
            min_sample=15
        )
        
        await _run(args)
        
        # Assert regime_core was called with the correct min_sample
        mock_regime_core.assert_called_once()
        assert mock_regime_core.call_args.kwargs["min_sample"] == 15
        
        # Assert exporter was called since neither symbol nor signal_id was passed
        mock_exporter.assert_called_once()
    asyncio.run(_do())


def test_regime_cli_symbol_case_study(mock_regime_core, mock_exporter, mock_print_case_study):
    async def _do():
        args = argparse.Namespace(
            generate=False,
            symbol=None,
            signal_id=None,
            summary=False,
            runner_threshold=2.0,
            regime=True,
            regime_symbol="GUN/USDT:USDT",
            regime_signal_id=None,
            min_sample=30
        )
        
        await _run(args)
        
        # Assert regime_core was called
        mock_regime_core.assert_called_once()
        
        # Assert exporter was NOT called
        mock_exporter.assert_not_called()
        
        # Assert case study was printed
        mock_print_case_study.assert_called_once()
        assert mock_print_case_study.call_args.args[2] == "GUN/USDT:USDT"
    asyncio.run(_do())


def test_regime_cli_signal_id_case_study(mock_regime_core, mock_exporter, mock_print_case_study):
    async def _do():
        args = argparse.Namespace(
            generate=False,
            symbol=None,
            signal_id=None,
            summary=False,
            runner_threshold=2.0,
            regime=True,
            regime_symbol=None,
            regime_signal_id="SIG_123",
            min_sample=30
        )
        
        await _run(args)
        
        # Assert regime_core was called
        mock_regime_core.assert_called_once()
        
        # Assert exporter was NOT called
        mock_exporter.assert_not_called()
        
        # Assert case study was printed
        mock_print_case_study.assert_called_once()
        assert mock_print_case_study.call_args.args[1] == "SIG_123"
    asyncio.run(_do())

def test_regime_args_parsed(monkeypatch):
    test_args = ["__main__.py", "--regime", "--regime-symbol", "GUNUSDT", "--min-sample", "10"]
    monkeypatch.setattr("sys.argv", test_args)
    
    with patch("asyncio.run") as mock_asyncio_run:
        main()
        
        # We just verify it parses the args and passes them without raising SystemExit
        pass

