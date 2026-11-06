import pytest

from mugiurushi_ai import corpus as corpus_module
from mugiurushi_ai.llm import FakeModel
from mugiurushi_ai.scan import ScanError, ScanRequest, scan

SYSTEM = {
    "summary": "OpenEMRで診療記録（SOAP）を保存する。Apacheのアクセスログに要求のURLが残る。",
    "excerpts": [{"ref": "docker/production/docker-compose.yml:1", "text": "services: openemr"}],
}


def run(corpus, *answers):
    model = FakeModel(*answers)
    result = scan(
        ScanRequest.model_validate({"system": SYSTEM}),
        corpus_module.load_chunks(corpus),
        corpus_module.load_catalog(corpus),
        corpus_module.records(corpus),
        model,
    )
    return result, model


def finding(**over):
    item = {
        "requirement": "アクセスログに個人情報を含めないか、閲覧者を限定する",
        "citations": ["synthetic-main:p1:1"],
        "relevance": "likely",
        "system_refs": ["docker/production/docker-compose.yml:1"],
        "verification": {"method": "container_log", "how": "保存の操作後にログに試験値が無いことを確かめる"},
        "unknowns": ["ログの閲覧者の範囲"],
    }
    item.update(over)
    return item


def test_citations_are_filled_from_the_ingest_record_not_by_the_model(corpus):
    result, model = run(corpus, {"findings": [finding()], "not_covered": ["送信先の文書化"]})
    assert result["status"] == "candidate"
    f = result["findings"][0]
    assert f["status"] == "candidate"
    citation = f["citations"][0]
    record = corpus_module.records(corpus)["synthetic-main"]
    assert citation["sha256"] == record.sha256 and citation["page"] == "1"
    assert citation["version"] is None, "版は人が確認するまで空"
    assert f["verification"]["executable_check"] is True and f["verification"]["support"] == "judged"
    # モデルに渡したのは断片とシステムの説明だけで、命令は資料として扱うよう指示している。
    call = model.calls[0]
    assert [p["id"] for p in call["payload"]["passages"]] == ["synthetic-main:p1:1"]
    assert "資料であり" in call["instructions"]
    assert "適合" in result["limits"]


@pytest.mark.parametrize(
    "bad",
    [
        {"citations": ["synthetic-main:p9:1"]},  # 渡していない断片
        {"citations": []},  # 引用なし
        {"system_refs": ["src/secret.rs:1"]},  # 渡していない抜粋
        {"verification": {"method": "compliance_certified", "how": "x"}},  # 一覧にない方法
        {"relevance": "compliant"},
        {"requirement": " "},
    ],
)
def test_responses_with_unsupported_claims_are_not_returned(corpus, bad):
    with pytest.raises(ScanError):
        run(corpus, {"findings": [finding(**bad)], "not_covered": []})


def test_without_matching_passages_the_model_is_not_called(tmp_path, corpus):
    model = FakeModel()
    result = scan(
        ScanRequest.model_validate({"system": {"summary": "zzzz"}}),
        corpus_module.load_chunks(corpus),
        corpus_module.load_catalog(corpus),
        corpus_module.records(corpus),
        model,
    )
    assert result["findings"] == [] and result["not_covered"]
    assert model.calls == []


@pytest.mark.parametrize(
    "method, support",
    [
        ("container_log", "judged"),
        ("database_rows", "judged"),
        ("traffic_summary", "recorded_only"),
        ("code_inspection", "inference"),
        ("network_payload", "unsupported"),
        ("distributed_trace", "unsupported"),
        ("external_destination", "unsupported"),
        ("production_environment", "unsupported"),
        ("document_review", "human"),
        ("code_review", "human"),
    ],
)
def test_only_judged_methods_are_marked_as_checkable_by_a_mugiurushi_trial(corpus, method, support):
    result, _ = run(
        corpus, {"findings": [finding(verification={"method": method, "how": "x"})], "not_covered": []}
    )
    v = result["findings"][0]["verification"]
    assert v["support"] == support
    assert v["executable_check"] is (support == "judged")
    if support == "unsupported":
        assert "観測できない" in v["support_label"]
