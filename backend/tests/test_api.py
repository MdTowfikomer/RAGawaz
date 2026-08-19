import os
os.environ["MAX_PASSAGES"] = "100"

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    """Verify health endpoint returns status 200 and specs."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "spec_version" in data


def test_api_query_endpoint(client):
    """Verify query endpoint returns grounded response and telemetry."""
    payload = {"query": "भारत की राजधानी क्या है?"}
    resp = client.post("/api/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "status" in data
    assert "telemetry" in data
    assert "embed_retrieval_ms" in data["telemetry"]
    assert "harness_ms" in data["telemetry"]


def test_api_benchmark_results_endpoint(client):
    """Verify benchmark results can be fetched."""
    resp = client.get("/api/benchmark/results")
    assert resp.status_code == 200
    data = resp.json()
    assert "stage_1" in data or "message" in data