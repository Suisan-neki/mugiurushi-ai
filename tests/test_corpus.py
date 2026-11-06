import hashlib
import json

import pytest

from mugiurushi_ai import corpus as corpus_module
from mugiurushi_ai.retrieve import search


def test_ingest_records_the_hash_and_pages_of_the_file_actually_read(corpus):
    record = corpus_module.records(corpus)["synthetic-main"]
    data = (corpus / "synthetic.md").read_bytes()
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    assert record.bytes == len(data) and record.pages == 1 and record.chunks >= 1
    assert record.origin.startswith("file:synthetic.md")
    chunks = corpus_module.load_chunks(corpus)
    assert all(c.id.startswith("synthetic-main:p1:") for c in chunks)
    assert "アクセスログ" in "".join(c.text for c in chunks)


def test_catalog_lists_only_documents_and_never_fills_versions(tmp_path):
    from pathlib import Path

    catalog = corpus_module.load_catalog(Path(__file__).resolve().parents[1] / "corpus")
    ids = [d.id for s in catalog.sources for d in s.documents]
    assert len(ids) == len(set(ids))
    # 版は取り込んだ文書を人が確認して記入する。一覧の時点では空。
    assert all(d.version is None for s in catalog.sources for d in s.documents)
    assert all(d.url or d.git for s in catalog.sources for d in s.documents)


def test_split_keeps_each_part_within_the_limit():
    parts = corpus_module.split("あ" * 3000 + "\n\n" + "い" * 10, size=1200)
    assert all(len(p) <= 1200 for p in parts)
    assert "".join(parts).replace("\n", "") == "あ" * 3000 + "い" * 10


def test_unsupported_file_types_are_rejected(corpus):
    other = corpus / "x.docx"
    other.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError):
        corpus_module.ingest(corpus, "synthetic-main", other)


def test_search_ranks_japanese_passages_by_overlap(corpus):
    chunks = corpus_module.load_chunks(corpus)
    found = search(chunks, "アクセスログに患者の氏名が残らないか", limit=1)
    assert found and "アクセスログ" in found[0].text
    assert search(chunks, "zzzz", limit=3) == []
    assert json.loads(json.dumps([c.id for c in found]))


def test_spaces_inserted_between_japanese_characters_by_pdf_extraction_are_removed():
    assert corpus_module.normalize("医 療 ・ 介 護 関 係 事 業 者") == "医療・介護関係事業者"
    assert corpus_module.normalize("NIST SP 800-53 の 管理策") == "NIST SP 800-53 の管理策"
    assert corpus_module.split("第 7.0 版\n概 説 編") == ["第 7.0 版 概説編"]
