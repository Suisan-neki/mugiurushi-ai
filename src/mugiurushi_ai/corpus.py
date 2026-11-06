"""照合に使う文書の一覧（catalog.json）と、取り込んだ断片。

取り込みは公開元（またはローカルのファイル）から行い、取得日・SHA-256・ページを記録する。
版の名称は取り込んだ文書を人が確認して記入し、ここでは推測しない。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

CHUNK_CHARS = 1200
SNAPSHOT = "snapshot.json"
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024


class GitLocation(BaseModel):
    repository: str
    ref: str
    path: str


class Document(BaseModel):
    id: str
    listed_title: str
    url: str | None = None
    git: GitLocation | None = None
    checked_on: str
    version: str | None = None
    fetch_note: str | None = None


class Source(BaseModel):
    id: str
    title: str
    publisher: str
    kind: str
    page_url: str | None = None
    license_note: str
    documents: list[Document] = Field(default_factory=list)


class Catalog(BaseModel):
    schema_version: int
    note: str
    sources: list[Source]

    def find(self, document_id: str) -> tuple[Source, Document]:
        for source in self.sources:
            for document in source.documents:
                if document.id == document_id:
                    return source, document
        raise KeyError(f"一覧にない文書です: {document_id}")


def load_catalog(corpus: Path) -> Catalog:
    return Catalog.model_validate_json((corpus / "catalog.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Chunk:
    """文書の一部。id は `文書ID:p{ページ}:{連番}`。page は PDF のページ番号（1始まり）か、Markdownのファイル名。"""

    id: str
    source_id: str
    document_id: str
    page: str
    text: str


@dataclass(frozen=True)
class IngestRecord:
    document_id: str
    source_id: str
    origin: str
    sha256: str
    bytes: int
    fetched_at: str
    pages: int
    chunks: int
    first_lines: str


# 和文（かな・漢字・全角記号）。PDFの文字抽出では、和文の文字の間に空白が入ることがある。
_CJK = "\u3000-\u30ff\u3400-\u9fff\uf900-\ufaff\uff00-\uffef"
_CJK_GAP = re.compile(f"(?<=[{_CJK}])[ \t\u3000]+(?=[{_CJK}])")


def normalize(text: str) -> str:
    """和文の文字どうしの間の空白（改行以外）を詰める。英数字の語の間の空白は残す。"""
    return _CJK_GAP.sub("", text)


def split(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """空白をまとめ、段落の切れ目を優先して size 文字以内に分ける。"""
    text = normalize(text)
    paragraphs = [re.sub(r"[ \t　]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    parts: list[str] = []
    current = ""
    for paragraph in filter(None, paragraphs):
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        while len(paragraph) > size:
            if current:
                parts.append(current)
                current = ""
            parts.append(paragraph[:size])
            paragraph = paragraph[size:]
        if len(current) + len(paragraph) + 1 > size and current:
            parts.append(current)
            current = ""
        current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]


def fetch_url(url: str) -> bytes:
    # 環境のプロキシと証明書の設定に従う。リダイレクトは公開元の同じ文書への移動に限らないため追わない。
    with httpx.Client(timeout=120, follow_redirects=False, trust_env=True) as client:
        response = client.get(url)
    if response.status_code != 200:
        raise OSError(f"{url} を取得できません（HTTP {response.status_code}）")
    if len(response.content) > MAX_DOCUMENT_BYTES:
        raise OSError(f"{url} が上限（{MAX_DOCUMENT_BYTES}バイト）を超えます")
    return response.content


def git_markdown(location: GitLocation) -> list[tuple[str, str]]:
    """指定したタグまたはコミット（40桁の16進）の、指定ディレクトリの Markdown を（ファイル名、本文）で返す。"""
    with tempfile.TemporaryDirectory() as directory:
        run = lambda *args: subprocess.run(args, check=True, capture_output=True, timeout=600)  # noqa: E731
        if re.fullmatch(r"[0-9a-f]{40}", location.ref):
            # タグのない取得元は、コミットを指定して固定する。
            run("git", "init", "-q", directory)
            run("git", "-C", directory, "remote", "add", "origin", location.repository)
            run("git", "-C", directory, "sparse-checkout", "set", location.path)
            run(
                "git",
                "-C",
                directory,
                "fetch",
                "-q",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                location.ref,
            )
            run("git", "-C", directory, "checkout", "-q", "FETCH_HEAD")
        else:
            run(
                "git",
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                location.ref,
                "--filter=blob:none",
                "--sparse",
                location.repository,
                directory,
            )
            run("git", "-C", directory, "sparse-checkout", "set", location.path)
        root = Path(directory) / location.path
        return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.md"))]


def ingest(corpus: Path, document_id: str, file: Path | None = None) -> IngestRecord:
    """文書を取り込み、`corpus/data/<文書ID>/` に記録と断片を保存する。"""
    catalog = load_catalog(corpus)
    source, document = catalog.find(document_id)
    pages: list[tuple[str, str]]
    if file is not None:
        data = file.read_bytes()
        origin = f"file:{file.name}（運営者が取得したファイル。取得元: {document.url or document.git}）"
        pages = _pages_of(file.name, data)
    elif document.url:
        data = fetch_url(document.url)
        origin = document.url
        pages = _pages_of(document.url, data)
    elif document.git:
        files = git_markdown(document.git)
        data = json.dumps(files, ensure_ascii=False).encode()
        origin = f"{document.git.repository}@{document.git.ref}:{document.git.path}"
        pages = files
    else:
        raise ValueError(f"取得先のない文書です: {document_id}")
    chunks = [
        Chunk(f"{document.id}:p{page}:{n}", source.id, document.id, page, text)
        for page, page_text in pages
        for n, text in enumerate(split(page_text), start=1)
    ]
    first = next((t for _, t in pages if t.strip()), "")
    record = IngestRecord(
        document_id=document.id,
        source_id=source.id,
        origin=origin,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pages=len(pages),
        chunks=len(chunks),
        first_lines=" ".join(first.split())[:300],
    )
    # 取得記録と断片を一つのファイルにまとめ、一時ファイルから置き換える。読む側は、取得記録と断片が
    # 別の版になった状態を見ない。ここまでで失敗した場合は、以前の有効な版がそのまま残る。
    out = corpus / "data" / document.id
    out.mkdir(parents=True, exist_ok=True)
    # 一時ファイルは取り込みごとに別の名前にする（同じプロセスで並行して取り込んでも混ざらない）。
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=out, prefix=f".{SNAPSHOT}.", suffix=".tmp", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        temporary.write_text(
            json.dumps({"record": asdict(record), "chunks": [asdict(c) for c in chunks]}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, out / SNAPSHOT)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def _pages_of(name: str, data: bytes) -> list[tuple[str, str]]:
    if data[:5] == b"%PDF-":
        return [(str(i), text) for i, text in enumerate(pdf_pages(data), start=1)]
    if name.endswith((".md", ".txt")):
        return [("1", data.decode("utf-8"))]
    raise ValueError(f"PDF・Markdown・テキスト以外は取り込めません: {name}")


@dataclass(frozen=True)
class Snapshot:
    """一つの文書の、ある時点の取得記録と断片。必ず同じファイルから一緒に読む。"""

    record: IngestRecord
    chunks: tuple[Chunk, ...]


def load_snapshots(corpus: Path) -> dict[str, Snapshot]:
    """取り込み済みの文書を、文書ごとに一度の読み込みで取得記録と断片を一緒に読む。"""
    found = {}
    for path in sorted((corpus / "data").glob(f"*/{SNAPSHOT}")):
        data = json.loads(path.read_text(encoding="utf-8"))
        record = IngestRecord(**data["record"])
        chunks = tuple(Chunk(**c) for c in data["chunks"])
        if len(chunks) != record.chunks or any(c.document_id != record.document_id for c in chunks):
            raise ValueError(f"取り込みの記録と断片が一致しません: {path}")
        found[record.document_id] = Snapshot(record, chunks)
    return found


def records(corpus: Path) -> dict[str, IngestRecord]:
    return {k: s.record for k, s in load_snapshots(corpus).items()}


def chunks_of(snapshots: dict[str, Snapshot], source_ids: list[str] | None = None) -> list[Chunk]:
    return [
        c for s in snapshots.values() for c in s.chunks if source_ids is None or c.source_id in source_ids
    ]


def load_chunks(corpus: Path, source_ids: list[str] | None = None) -> list[Chunk]:
    return chunks_of(load_snapshots(corpus), source_ids)
