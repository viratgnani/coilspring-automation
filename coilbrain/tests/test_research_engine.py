"""Tests for the Research Engine V1."""

import pytest
import time

from coilbrain.research import engine

def test_get_horizon_tolerance():
    # max(60, horizon * 0.1)
    assert engine._get_horizon_tolerance(30) == 60.0
    assert engine._get_horizon_tolerance(60) == 60.0
    assert engine._get_horizon_tolerance(120) == 60.0
    assert engine._get_horizon_tolerance(300) == 60.0
    assert engine._get_horizon_tolerance(600) == 60.0
    assert engine._get_horizon_tolerance(1800) == 180.0
    assert engine._get_horizon_tolerance(3600) == 360.0


def test_safe_pct_change():
    assert engine._safe_pct_change(100.0, 110.0) == 10.0
    assert engine._safe_pct_change(100.0, 50.0) == -50.0
    assert engine._safe_pct_change(None, 110.0) is None
    assert engine._safe_pct_change(100.0, None) is None
    assert engine._safe_pct_change(None, None) is None
    assert engine._safe_pct_change(0.0, 10.0) is None


def test_safe_delta():
    assert engine._safe_delta(100.0, 110.0) == 10.0
    assert engine._safe_delta(100.0, 50.0) == -50.0
    assert engine._safe_delta(None, 110.0) is None
    assert engine._safe_delta(100.0, None) is None
    assert engine._safe_delta(None, None) is None
    assert engine._safe_delta(0.0, 10.0) == 10.0
