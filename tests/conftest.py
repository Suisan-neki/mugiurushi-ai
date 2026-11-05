"""架空の文書を取り込んだ一時的なコーパス。実在のガイドラインの本文は使わない。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mugiurushi_ai import corpus as corpus_module

SYNTHETIC = """# 合成の安全管理ガイド（試験用。実在の文書ではない）

診療録を保存するシステムは、保存した記録を利用者ごとに区別し、ほかの患者の記録と混ざらないようにする。

アクセスログには、患者の氏名や生年月日などの個人情報を含めないか、含める場合は閲覧できる者を限定する。

外部のサービスへ診療情報を送信する場合は、送信先と送信する項目を文書で定める。
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    catalog = {
        "schema_version": 1,
        "note": "試験用",
        "sources": [
            {
                "id": "synthetic",
                "title": "合成の安全管理ガイド",
                "publisher": "試験",
                "kind": "domestic_guideline",
                "page_url": None,
                "license_note": "試験用の架空の文書",
                "documents": [
                    {
                        "id": "synthetic-main",
                        "listed_title": "合成の安全管理ガイド（試験用）",
                        "url": None,
                        "checked_on": "2026-11-05",
                        "version": None,
                    }
                ],
            }
        ],
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    source = tmp_path / "synthetic.md"
    source.write_text(SYNTHETIC, encoding="utf-8")
    corpus_module.ingest(tmp_path, "synthetic-main", source)
    return tmp_path
