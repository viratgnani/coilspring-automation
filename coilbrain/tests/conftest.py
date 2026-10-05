"""Shared pytest configuration for the CoilBrain test suite.

Sets up a session-scoped temporary SQLite database before any test module
is imported, so the TestClient lifespan picks up the correct database_url.
"""

import asyncio
import pathlib
import pytest

from coilbrain.config import settings
from coilbrain import database


@pytest.fixture(scope="session", autouse=True)
def _session_db(tmp_path_factory):
    """Create a temp SQLite DB once for the entire test session.

    This fixture runs before the TestClient is created, ensuring
    settings.database_url is set to the temp file before main.py''s
    lifespan calls database.init_db().
    """
    db_file = tmp_path_factory.mktemp("session_db") / "coilbrain_test.db"
    settings.database_url = str(db_file)
    # Init schema immediately so observatory tests can also use it directly.
    asyncio.run(database.init_db(str(db_file)))
    yield str(db_file)
