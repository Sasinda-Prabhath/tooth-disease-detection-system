import pytest

fastapi = pytest.importorskip("fastapi")
tensorflow = pytest.importorskip("tensorflow")

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "module2-impaction-boneloss"
