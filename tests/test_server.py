import pytest
from fastapi.testclient import TestClient

from mugiurushi_ai.llm import FakeModel
from mugiurushi_ai.server import create_app

TOKEN = "t" * 32


def client(corpus, model=None):
    return TestClient(create_app(token=TOKEN, corpus=corpus, model=model, model_note="未設定の理由"))


def test_token_is_required_and_short_tokens_are_refused(corpus):
    c = client(corpus)
    assert c.get("/v1/health").status_code == 403
    assert c.get("/v1/health", headers={"X-Mugiurushi-AI-Token": "wrong"}).status_code == 403
    ok = c.get("/v1/health", headers={"X-Mugiurushi-AI-Token": TOKEN})
    assert ok.status_code == 200 and ok.json()["model_status"] == "not_configured"
    assert ok.json()["ingested_documents"] == ["synthetic-main"]
    with pytest.raises(ValueError):
        create_app(token="short", corpus=corpus, model=None)


def test_without_a_model_the_service_says_so_instead_of_answering(corpus):
    r = client(corpus).post(
        "/v1/dialogue/turn",
        headers={"X-Mugiurushi-AI-Token": TOKEN},
        json={"system": "s", "concern": "c"},
    )
    assert r.status_code == 503 and "未設定" in r.json()["detail"]


def test_large_bodies_are_rejected(corpus):
    r = client(corpus).post(
        "/v1/scan",
        headers={"X-Mugiurushi-AI-Token": TOKEN, "Content-Type": "application/json"},
        content=b"{" + b" " * (1024 * 1024) + b"}",
    )
    assert r.status_code == 413


def test_invalid_model_output_is_a_bad_gateway_not_a_result(corpus):
    bad = {
        "findings": [
            {
                "requirement": "x",
                "citations": ["nope"],
                "relevance": "likely",
                "system_refs": [],
                "verification": {"method": "container_log", "how": "x"},
                "unknowns": [],
            }
        ],
        "not_covered": [],
    }
    r = client(corpus, FakeModel(bad)).post(
        "/v1/scan",
        headers={"X-Mugiurushi-AI-Token": TOKEN},
        json={"system": {"summary": "アクセスログに患者の氏名が残らないか"}},
    )
    assert r.status_code == 502


def test_sources_show_ingest_records(corpus):
    r = client(corpus).get("/v1/sources", headers={"X-Mugiurushi-AI-Token": TOKEN})
    doc = r.json()["sources"][0]["documents"][0]
    assert doc["ingested"]["sha256"] and doc["version"] is None
