"""Tests for Research Analysis V1."""

import pytest
from coilbrain.research.analysis import core

def test_aggregate_empty_returns_low_sample():
    res = core._aggregate("test", [], min_sample=30)
    assert res.bucket_name == "test"
    assert res.n == 0
    assert res.unique_symbols == 0
    assert res.sample_quality == "LOW_SAMPLE"
    assert res.mean_forward_return is None
    assert res.median_forward_return is None


def test_aggregate_low_sample():
    vals = [("SYM1", 1.0), ("SYM2", -1.0)]
    res = core._aggregate("test", vals, min_sample=30)
    assert res.n == 2
    assert res.unique_symbols == 2
    assert res.sample_quality == "LOW_SAMPLE"
    assert res.mean_forward_return == 0.0
    assert res.median_forward_return == 0.0
    assert res.positive_return_pct == 50.0


def test_aggregate_ok_sample():
    vals = [(f"SYM{i}", 1.0) for i in range(30)]
    res = core._aggregate("test", vals, min_sample=30)
    assert res.n == 30
    assert res.unique_symbols == 30
    assert res.sample_quality == "OK"
    assert res.mean_forward_return == 1.0
    assert res.positive_return_pct == 100.0


def test_aggregate_repeated_symbols():
    vals = [("SYM1", 1.0), ("SYM1", 2.0), ("SYM2", 1.0)]
    res = core._aggregate("test", vals, min_sample=3)
    assert res.n == 3
    assert res.unique_symbols == 2
    assert res.sample_quality == "OK"
    assert res.mean_forward_return == (1.0 + 2.0 + 1.0) / 3.0
