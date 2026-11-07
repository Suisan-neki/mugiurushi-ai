import pytest

from mugiurushi_ai import corpus as corpus_module
from mugiurushi_ai.flow import AskRequest, FlowError, PlanRequest, ask, plan
from mugiurushi_ai.llm import FakeModel
from mugiurushi_ai.scan import ScanRequest, priority, scan

SYSTEM = {
    "summary": "架空の診療所のタスク管理。職員がログインして患者ごとのタスクを追加する。",
    "excerpts": [{"ref": "app/auth.py", "text": "def login(): ..."}],
}


def test_priority_is_a_fixed_rule_from_relevance_and_patient_data():
    assert priority("likely", True) == "high"
    assert priority("likely", False) == "medium"
    assert priority("possible", True) == "medium"
    assert priority("unclear", False) == "low"


def test_scan_returns_hearing_questions_and_rule_based_priority(corpus):
    model = FakeModel(
        {
            "findings": [
                {
                    "requirement": "アクセスログに個人情報を含めない",
                    "citations": ["synthetic-main:p1:1"],
                    "relevance": "likely",
                    "system_refs": [],
                    "verification": {"method": "container_log", "how": "ログを読む"},
                    "unknowns": [],
                    "patient_data": True,
                }
            ],
            "not_covered": [],
            "questions": [
                {
                    "text": "職員ごとに担当患者は決まっていますか",
                    "why": "閲覧範囲の試験に使う",
                    "options": ["はい", "いいえ"],
                }
            ],
        }
    )
    request = ScanRequest.model_validate(
        {"system": SYSTEM, "answers": [{"question": "院外から使いますか", "answer": "院内だけ"}]}
    )
    result = scan(request, corpus_module.load_snapshots(corpus), corpus_module.load_catalog(corpus), model)
    assert result["findings"][0]["priority"] == "high" and result["findings"][0]["patient_data"] is True
    assert result["findings"][0]["citations"][0]["source_id"] == "synthetic"
    assert result["questions"] == [
        {
            "id": "q1",
            "text": "職員ごとに担当患者は決まっていますか",
            "why": "閲覧範囲の試験に使う",
            "options": ["はい", "いいえ"],
        }
    ]
    assert model.calls[0]["payload"]["answers"] == [{"question": "院外から使いますか", "answer": "院内だけ"}]


def test_ask_answers_only_from_passages_and_returns_citations(corpus):
    snapshots, catalog = corpus_module.load_snapshots(corpus), corpus_module.load_catalog(corpus)
    model = FakeModel({"answer": "閲覧できる者を限定します。", "citations": ["synthetic-main:p1:1"]})
    result = ask(AskRequest(question="アクセスログに個人情報を含めてよいですか"), snapshots, catalog, model)
    assert (
        result["grounded"]
        and result["citations"][0]["sha256"] == corpus_module.records(corpus)["synthetic-main"].sha256
    )
    with pytest.raises(FlowError):
        ask(
            AskRequest(question="アクセスログ"),
            snapshots,
            catalog,
            FakeModel({"answer": "x", "citations": ["made-up"]}),
        )


def finding_ids(*ids):
    return [{"id": i, "requirement": f"要求{i}"} for i in ids]


def test_plan_groups_findings_and_rejects_unknown_or_duplicated_ones():
    good = {
        "tests": [
            {
                "name": "別の職員での操作",
                "kind": "browser",
                "finding_ids": ["f1", "f2"],
                "steps": ["職員2でログインする"],
                "pass": ["表示されない"],
                "observation": "browser_text",
            },
        ],
        "not_planned": [{"finding_id": "f3", "reason": "本番の構成による"}],
    }
    result = plan(
        PlanRequest.model_validate({"system": SYSTEM, "findings": finding_ids("f1", "f2", "f3")}),
        FakeModel(good),
    )
    assert (
        result["tests"][0]["finding_ids"] == ["f1", "f2"] and result["not_planned"][0]["finding_id"] == "f3"
    )
    assert result["status"] == "draft"
    for bad in [
        {**good["tests"][0], "finding_ids": ["f9"]},
        {**good["tests"][0], "observation": "code_inspection"},
        {**good["tests"][0], "steps": []},
    ]:
        with pytest.raises(FlowError):
            plan(
                PlanRequest.model_validate({"system": SYSTEM, "findings": finding_ids("f1", "f2")}),
                FakeModel({"tests": [bad], "not_planned": []}),
            )
    with pytest.raises(FlowError):
        plan(
            PlanRequest.model_validate({"system": SYSTEM, "findings": finding_ids("f1", "f2")}),
            FakeModel(
                {
                    "tests": [good["tests"][0], {**good["tests"][0], "name": "別", "finding_ids": ["f2"]}],
                    "not_planned": [],
                }
            ),
        )
