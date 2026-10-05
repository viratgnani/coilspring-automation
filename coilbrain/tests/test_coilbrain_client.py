"""CoilBrain-suite stub: V2 connector tests relocated.

``coilbrain_client.py`` belongs to the CoilSpring V2 project
(``coiled_spring_scanner.py``).  Its tests were written against an earlier
API signature::

    build_payload(result, bundle, cfg, timestamp)   # OLD — removed
    notify(result, bundle, cfg)                     # OLD — removed
    _post(url, payload)                             # OLD — removed

The current V2 connector API is::

    build_payload(result)          # derives all fields from AnalysisResult
    notify(result)                 # reads COILBRAIN_URL from environment
    _post(url, payload, symbol)    # logs symbol on success

Tests for the current API are maintained in the V2 project::

    test_v2_coilbrain_client.py   (workspace root / /home/ubuntu/ on VPS)

To run V2 connector tests:
    pytest test_v2_coilbrain_client.py -v            # from workspace root
    pytest /home/ubuntu/test_v2_coilbrain_client.py  # on VPS

This file is intentionally skipped within the CoilBrain test suite to
prevent cross-project coupling and stale-API failures.
"""
import pytest

pytest.skip(
    reason=(
        "V2 connector tests belong in the CoilSpring V2 project. "
        "Current API tests live in test_v2_coilbrain_client.py "
        "(workspace root / /home/ubuntu/ on VPS)."
    ),
    allow_module_level=True,
)
