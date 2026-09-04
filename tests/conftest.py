import pytest
from src.analytics.store import _cleanup_all_stores


@pytest.fixture(scope="session", autouse=True)
def cleanup_stores_on_session_exit():
    """Ensures all DuckDBAnalyticsStore background threads and connections are joined before pytest process exit."""
    yield
    _cleanup_all_stores()
