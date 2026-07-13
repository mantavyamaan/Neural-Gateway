import os
import pytest
import tempfile
import sqlite3

# Force all tests to use a temporary SQLite file
fd, temp_db_path = tempfile.mkstemp(suffix=".db")
os.close(fd)
os.environ["ATLAS_DB_PATH"] = temp_db_path

from app.core.database import init_db, get_all_models, upsert_model
from app.models.registry_builder import build_registry

@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    init_db()
    if not get_all_models():
        models = build_registry()
        for m in models:
            upsert_model(m)
    yield
    # Cleanup after tests
    try:
        os.remove(temp_db_path)
    except OSError:
        pass
