"""実際のOpenAI APIを使う試験。既定では実行しない（MUGIURUSHI_AI_LIVE=1 のときだけ）。

決定的な試験と分ける。モデルの応答の内容は確かめず、サーバー側の検証を通った形で返るかだけを見る。
"""

import os
from pathlib import Path

import pytest

from mugiurushi_ai import corpus as corpus_module
from mugiurushi_ai.config import ModelConfig
from mugiurushi_ai.dialogue import TurnRequest, turn
from mugiurushi_ai.llm import OpenAIModel
from mugiurushi_ai.scan import ScanRequest, scan

pytestmark = pytest.mark.skipif(
    os.environ.get("MUGIURUSHI_AI_LIVE") != "1", reason="MUGIURUSHI_AI_LIVE=1 のときだけ"
)
CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def model():
    config = ModelConfig.from_env()
    assert config, "MUGIURUSHI_AI_MODEL を設定してください"
    return OpenAIModel(config)


def test_scan_with_the_real_model_returns_only_validated_citations():
    if not corpus_module.records(CORPUS):
        pytest.skip("文書が取り込まれていない（mugiurushi-ai ingest --all）")
    result = scan(
        ScanRequest.model_validate(
            {
                "system": {
                    "summary": "OpenEMRで診療記録（SOAP）を保存し、再表示する。Apacheのアクセスログに要求のURLが残る。"
                }
            }
        ),
        corpus_module.load_chunks(CORPUS),
        corpus_module.load_catalog(CORPUS),
        corpus_module.records(CORPUS),
        model(),
    )
    assert result["status"] == "candidate"
    passages = {p["id"] for p in result["passages"]}
    assert all(c["passage_id"] in passages for f in result["findings"] for c in f["citations"])


def test_dialogue_with_the_real_model_returns_unlinked_drafts():
    result = turn(
        TurnRequest.model_validate(
            {"system": "OpenEMRでSOAPを保存する", "concern": "診療データは意図した場所に正しく保存される？"}
        ),
        model(),
    )
    assert result["status"] == "draft"
    assert all(a["aspect"]["trial_checks"] == [] for a in result["aspects"])
