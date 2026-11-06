"""実際のOpenAI APIを使う試験。既定では実行しない（MUGIURUSHI_AI_LIVE=1 のときだけ）。

決定的な試験と分ける。送るのは公開文書の断片と架空のデータだけ。応答の内容の良し悪しは判定せず、
①照合・②試験範囲の案・②結果の説明の3経路が、サーバーの検証を通った形で1件以上返ることを確かめる。
"""

import json
import os
from pathlib import Path

import pytest

from mugiurushi_ai import corpus as corpus_module
from mugiurushi_ai.config import ModelConfig
from mugiurushi_ai.dialogue import ExplainRequest, TurnRequest, explain, turn
from mugiurushi_ai.llm import OpenAIModel
from mugiurushi_ai.scan import ScanRequest, scan

pytestmark = pytest.mark.skipif(
    os.environ.get("MUGIURUSHI_AI_LIVE") != "1", reason="MUGIURUSHI_AI_LIVE=1 のときだけ"
)
CORPUS = Path(__file__).resolve().parents[2] / "corpus"
OUT = os.environ.get("MUGIURUSHI_AI_LIVE_OUT")

# 架空のシステムの説明。実在の医療機関・患者の情報は含まない。
SYSTEM = (
    "架空の小規模クリニック向けの電子カルテ。医師が診療記録（SOAP）を入力して保存し、後から開き直して読む。"
    "保存先はアプリと同じ試験用のMariaDB。Apacheのアクセスログに要求のURLが残る。"
)


def model():
    config = ModelConfig.from_env()
    assert config, "MUGIURUSHI_AI_MODEL を設定してください"
    return OpenAIModel(config)


def keep(name, result):
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def test_scan_returns_at_least_one_validated_candidate():
    if not corpus_module.records(CORPUS):
        pytest.skip("文書が取り込まれていない（mugiurushi-ai ingest --all）")
    result = scan(
        ScanRequest.model_validate(
            {"system": {"summary": SYSTEM}, "source_ids": ["mhlw-security", "ppc-medical-guidance"]}
        ),
        corpus_module.load_chunks(CORPUS),
        corpus_module.load_catalog(CORPUS),
        corpus_module.records(CORPUS),
        model(),
    )
    keep("scan", result)
    assert result["status"] == "candidate"
    assert len(result["findings"]) >= 1
    passages = {p["id"] for p in result["passages"]}
    assert all(c["passage_id"] in passages for f in result["findings"] for c in f["citations"])
    assert result["model_call"]["response_id"]


def test_dialogue_returns_at_least_one_unlinked_draft():
    result = turn(
        TurnRequest.model_validate(
            {"system": SYSTEM, "concern": "診療データは意図した場所に正しく保存される？"}
        ),
        model(),
    )
    keep("turn", result)
    assert result["status"] == "draft"
    assert len(result["aspects"]) >= 1
    assert all(a["aspect"]["trial_checks"] == [] for a in result["aspects"])


def test_explain_returns_one_explanation_per_aspect_and_keeps_statuses():
    request = {
        "check_text": "架空の患者のSOAPを保存し、開き直しても同じ内容を読める",
        "verdict": "satisfied",
        "verification_passed": False,
        "coverage": {
            "status": "partial",
            "summary": "登録した2件のうち1件を観測した。残りは未確認",
            "aspects": [
                {
                    "id": "p1",
                    "text": "SOAPが試験用DBに保存される",
                    "observation_label": "DBの行",
                    "status": "observed",
                    "status_label": "麦漆が観測した",
                    "detail": "判定条件 soap-row: 期待どおり",
                },
                {
                    "id": "p2",
                    "text": "診療データが外部へ送信されない",
                    "observation_label": "外部の送信先",
                    "status": "unsupported",
                    "status_label": "基盤が観測できない",
                    "detail": "外部の送信先は基盤が観測できない",
                },
            ],
        },
    }
    result = explain(ExplainRequest.model_validate(request), model())
    keep("explain", result)
    assert [a["id"] for a in result["aspects"]] == ["p1", "p2"]
    assert [a["status"] for a in result["aspects"]] == ["observed", "unsupported"]
    assert all(a["explanation"] for a in result["aspects"])
    assert result["verification_passed"] is False
