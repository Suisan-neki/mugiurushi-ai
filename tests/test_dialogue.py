import pytest

from mugiurushi_ai.dialogue import DialogueError, ExplainRequest, TurnRequest, explain, turn
from mugiurushi_ai.llm import FakeModel

TURN = {"system": "OpenEMRでSOAPを保存する", "concern": "診療データは意図した場所に正しく保存される？"}


def aspect(
    text="利用者がSOAPを保存すると、試験用DBの form_soap に同じ内容が保存される", observation="database_rows"
):
    return {"text": text, "observation": observation, "rationale": "保存先を直接読む", "unknowns": []}


def test_draft_aspects_have_no_links_and_match_mugiurushi_set_aspects(corpus):
    answer = {
        "reply": "保存先とログを分けて確かめます。外部送信は麦漆では観測できません。",
        "questions": ["本番の保存先はどこですか？"],
        "aspects": [aspect(), aspect("外部へ診療データが送信されない", "external_destination"), aspect()],
        "out_of_scope": [{"text": "本番環境の保存", "reason": "試験環境だけを扱う"}],
    }
    model = FakeModel(answer)
    result = turn(TurnRequest.model_validate(TURN), model)
    assert result["status"] == "draft"
    assert len(result["aspects"]) == 2, "同じ文の案は1件にまとめる"
    first = result["aspects"][0]["aspect"]
    assert set(first) == {"id", "text", "origin", "observation", "trial_checks"}
    assert first["trial_checks"] == [], "判定条件への対応づけは人が麦漆で選ぶ"
    assert "未確認" in first["origin"]
    assert result["aspects"][1]["support"] == "unsupported"
    kinds = {k["id"] for k in model.calls[0]["payload"]["observation_kinds"]}
    assert "database_rows" in kinds and len(kinds) == 22


@pytest.mark.parametrize("bad", [aspect(observation="telepathy"), aspect(text=" ")])
def test_unknown_kinds_and_empty_text_are_rejected(bad):
    with pytest.raises(DialogueError):
        turn(
            TurnRequest.model_validate(TURN),
            FakeModel({"reply": "x", "questions": [], "aspects": [bad], "out_of_scope": []}),
        )


def test_too_many_aspects_are_rejected():
    many = [aspect(text=f"部分{i}") for i in range(33)]
    with pytest.raises(DialogueError):
        turn(
            TurnRequest.model_validate(TURN),
            FakeModel({"reply": "x", "questions": [], "aspects": many, "out_of_scope": []}),
        )


COVERAGE = {
    "check_text": "SOAPを保存し、開き直しても同じ内容を読める",
    "verdict": "satisfied",
    "verification_passed": False,
    "coverage": {
        "status": "partial",
        "summary": "登録した2件のうち1件を観測した。残りは未確認",
        "aspects": [
            {
                "id": "p1",
                "text": "DBに保存される",
                "observation_label": "DBの行",
                "status": "observed",
                "status_label": "麦漆が観測した",
                "detail": "判定条件 soap-row: 期待どおり",
            },
            {
                "id": "p2",
                "text": "外部へ送信されない",
                "observation_label": "外部の送信先",
                "status": "unsupported",
                "status_label": "基盤が観測できない",
                "detail": "外部の送信先は基盤が観測できない",
            },
        ],
    },
}


def test_explanation_keeps_mugiurushi_statuses_and_only_adds_text():
    answer = {
        "summary": "保存は確認、外部送信は未確認",
        # 説明の文で「確認できた」と書いても、状態は麦漆の値のまま返す。
        "aspects": [
            {"id": "p1", "explanation": "DBの行を読み出した"},
            {"id": "p2", "explanation": "確認できた"},
        ],
        "next_steps": ["送信先を文書で確認する"],
    }
    result = explain(ExplainRequest.model_validate(COVERAGE), FakeModel(answer))
    assert result["verification_passed"] is False and result["verdict"] == "satisfied"
    assert [a["status"] for a in result["aspects"]] == ["observed", "unsupported"]
    assert "status" not in str(FakeModel(answer).calls)


@pytest.mark.parametrize(
    "items",
    [
        [{"id": "p1", "explanation": "x"}],  # 説明の抜け
        [{"id": "p1", "explanation": "x"}, {"id": "p3", "explanation": "y"}],  # 結果にない部分
        [{"id": "p1", "explanation": "x"}, {"id": "p1", "explanation": "y"}],  # 重複
    ],
)
def test_explanations_must_cover_exactly_the_given_aspects(items):
    answer = {"summary": "s", "aspects": items, "next_steps": []}
    with pytest.raises(DialogueError):
        explain(ExplainRequest.model_validate(COVERAGE), FakeModel(answer))
