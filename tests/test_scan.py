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
        corpus_module.load_snapshots(corpus),
        corpus_module.load_catalog(corpus),
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
        corpus_module.load_snapshots(corpus),
        corpus_module.load_catalog(corpus),
        model,
    )
    assert result["findings"] == [] and result["passages"] == []
    assert [s["status"] for s in result["sources"]] == ["not_selected"]
    assert result["sources"][0]["matched_passages"] == 0
    assert "モデルに渡していない" in result["sources"][0]["status_label"]
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


def test_citations_take_text_and_hash_from_the_same_snapshot_even_if_a_new_version_is_ingested(corpus):
    """照合の途中で同じ文書の新版が取り込まれても、旧版の本文に新版のSHA-256を付けない。"""
    old = corpus_module.load_snapshots(corpus)
    old_sha = old["synthetic-main"].record.sha256
    new_file = corpus / "v2.md"
    new_file.write_text(
        (corpus / "synthetic.md").read_text(encoding="utf-8").replace("アクセスログ", "アクセス記録（新版）"),
        encoding="utf-8",
    )
    new = corpus_module.ingest(corpus, "synthetic-main", new_file)
    assert new.sha256 != old_sha
    passage = old["synthetic-main"].chunks[0].id
    result = scan(
        ScanRequest.model_validate({"system": SYSTEM}),
        old,
        corpus_module.load_catalog(corpus),
        FakeModel({"findings": [finding(citations=[passage])], "not_covered": []}),
    )
    citation = result["findings"][0]["citations"][0]
    assert "アクセスログ" in citation["excerpt"] and citation["sha256"] == old_sha
    fresh = corpus_module.load_snapshots(corpus)["synthetic-main"]
    assert fresh.record.sha256 == new.sha256 and "新版" in fresh.chunks[0].text


def test_a_failed_ingest_keeps_the_previous_valid_version(corpus, monkeypatch):
    before = corpus_module.load_snapshots(corpus)["synthetic-main"].record.sha256
    bad = corpus / "x.docx"
    bad.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError):
        corpus_module.ingest(corpus, "synthetic-main", bad)
    # 置き換えの直前で失敗しても、以前の版が残り、一時ファイルも残らない。
    other = corpus / "v3.md"
    other.write_text("別の版", encoding="utf-8")

    def fail(*args, **kwargs):
        raise OSError("合成の書き込み失敗")

    monkeypatch.setattr(corpus_module.os, "replace", fail)
    with pytest.raises(OSError):
        corpus_module.ingest(corpus, "synthetic-main", other)
    monkeypatch.undo()
    assert corpus_module.load_snapshots(corpus)["synthetic-main"].record.sha256 == before
    assert not list((corpus / "data" / "synthetic-main").glob(".*.tmp"))


def test_each_requested_source_gets_a_server_side_status(corpus):
    import json

    catalog_path = corpus / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["sources"].append(
        {
            **catalog["sources"][0],
            "id": "not-yet",
            "documents": [
                {
                    "id": "not-yet-main",
                    "listed_title": "未取り込み",
                    "url": None,
                    "checked_on": "2026-11-05",
                    "version": None,
                }
            ],
        }
    )
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    result = scan(
        ScanRequest.model_validate({"system": SYSTEM, "source_ids": ["synthetic", "not-yet", "unknown"]}),
        corpus_module.load_snapshots(corpus),
        corpus_module.load_catalog(corpus),
        FakeModel({"findings": [finding()], "not_covered": []}),
    )
    statuses = {s["source_id"]: s for s in result["sources"]}
    assert statuses["synthetic"]["status"] == "used"
    assert statuses["synthetic"]["documents"][0]["passages_used"] >= 1
    assert statuses["not-yet"]["status"] == "not_ingested"
    assert statuses["unknown"]["status"] == "not_in_catalog"


def test_sources_outside_the_top_passages_are_not_said_to_have_no_matching_text(corpus):
    """上位の件数から外れた出典を「重なる断片が無い」と言わず、モデルに渡していないとだけ示す。"""
    result = scan(
        ScanRequest.model_validate({"system": SYSTEM, "max_passages": 1}),
        corpus_module.load_snapshots(corpus),
        corpus_module.load_catalog(corpus),
        FakeModel({"findings": [finding()], "not_covered": []}),
    )
    assert result["sources"][0]["status"] == "used"
    # 同じ出典の断片を2つにして、上位1件だけを渡す。
    import json

    catalog_path = corpus / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["sources"].append(
        {
            **catalog["sources"][0],
            "id": "second",
            "documents": [
                {
                    "id": "second-main",
                    "listed_title": "二つ目",
                    "url": None,
                    "checked_on": "2026-11-05",
                    "version": None,
                }
            ],
        }
    )
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    second = corpus / "second.md"
    second.write_text("アクセスログの取扱い（二つ目の合成文書）", encoding="utf-8")
    corpus_module.ingest(corpus, "second-main", second)
    result = scan(
        ScanRequest.model_validate({"system": SYSTEM, "max_passages": 1}),
        corpus_module.load_snapshots(corpus),
        corpus_module.load_catalog(corpus),
        _first_passage_model(),
    )
    statuses = {s["source_id"]: s for s in result["sources"]}
    left_out = [s for s in statuses.values() if s["status"] == "not_selected"]
    assert len(left_out) == 1 and left_out[0]["matched_passages"] >= 1
    assert "渡した1件に選ばれなかった" in left_out[0]["status_label"]
    assert "重なる断片が無かった" not in left_out[0]["status_label"]


class _first_passage_model:
    """渡された最初の断片を引用する試験用のモデル。"""

    name = "fake"

    def complete(self, *, instructions, payload, schema, name):
        passage = payload["passages"][0]["id"]
        return {"findings": [finding(citations=[passage], system_refs=[])], "not_covered": []}, {
            "response_id": "fake"
        }


def test_every_source_with_matching_text_gets_a_passage_before_the_rest_are_filled(tmp_path):
    """全体の順位で上位を占める出典があっても、重なる断片のある出典にはそれぞれ1件以上を渡す。"""
    from mugiurushi_ai.corpus import Chunk
    from mugiurushi_ai.retrieve import select

    ranked = [Chunk(f"a:p1:{n}", "a", "a-main", "1", "") for n in range(5)] + [
        Chunk("b:p1:1", "b", "b-main", "1", ""),
        Chunk("c:p1:1", "c", "c-main", "1", ""),
    ]
    assert [c.id for c in select(ranked, 4)] == ["a:p1:0", "a:p1:1", "b:p1:1", "c:p1:1"]
    assert [c.id for c in select(ranked, 2)] == ["a:p1:0", "b:p1:1"], (
        "出典が枠より多ければ、最上位の順位が低い出典が外れる"
    )
    assert [c.id for c in select(ranked, 10)] == [c.id for c in ranked]
