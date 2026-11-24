import pytest

from src import correction_memory


@pytest.fixture(autouse=True)
def _correction_memory_off(monkeypatch):
    """Some route tests use the default database path: never let them read real feedback.

    Tests of the correction memory turn it back on against their own database.
    """
    monkeypatch.setenv("NEXUS_ROUTER_CORRECTION_MEMORY", "off")
    correction_memory.invalidate()
    yield
    correction_memory.invalidate()
