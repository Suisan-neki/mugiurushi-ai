"""運営者が使うコマンド。文書の取り込み・一覧・サーバーの起動。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from . import corpus as corpus_module
from .config import ConfigError, ModelConfig, corpus_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mugiurushi-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="照合に使う文書の一覧と、取り込みの状況")
    ingest = sub.add_parser("ingest", help="文書を取り込む")
    ingest.add_argument("document_id", nargs="?")
    ingest.add_argument(
        "--all", action="store_true", help="一覧のすべての文書（取得できないものは理由を出して飛ばす）"
    )
    ingest.add_argument("--file", type=Path, help="運営者が取得したファイルから取り込む")
    serve = sub.add_parser("serve", help="HTTP APIを起動する")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    corpus = corpus_dir()

    if args.command == "sources":
        catalog = corpus_module.load_catalog(corpus)
        records = corpus_module.records(corpus)
        for source in catalog.sources:
            print(f"{source.id}  {source.title}（{source.publisher}）")
            for d in source.documents:
                state = (
                    f"取り込み済み {records[d.id].fetched_at} sha256={records[d.id].sha256[:12]}…"
                    if d.id in records
                    else "未取り込み"
                )
                print(f"  {d.id}  {state}  {d.listed_title}")
        return 0

    if args.command == "ingest":
        catalog = corpus_module.load_catalog(corpus)
        ids = [d.id for s in catalog.sources for d in s.documents] if args.all else [args.document_id]
        if not ids or ids == [None]:
            parser.error("文書のIDか --all を指定してください")
        failed = 0
        for document_id in ids:
            try:
                record = corpus_module.ingest(corpus, document_id, args.file)
                print(json.dumps(asdict(record), ensure_ascii=False))
            except Exception as error:  # 取得できない文書は理由を出して続ける
                failed += 1
                print(f"{document_id}: 取り込めません: {error}", file=sys.stderr)
        return 1 if failed else 0

    if args.command == "serve":
        import uvicorn

        from .llm import OpenAIModel
        from .server import create_app

        token = os.environ.get("MUGIURUSHI_AI_TOKEN", "")
        try:
            config = ModelConfig.from_env()
        except ConfigError as error:
            print(error, file=sys.stderr)
            return 2
        model = OpenAIModel(config) if config else None
        note = (
            ""
            if config
            else "運営者が MUGIURUSHI_AI_MODEL と認証（OPENAI_API_KEY または MUGIURUSHI_AI_AUTH=proxy）を設定する必要があります。"
        )
        try:
            app = create_app(token=token, corpus=corpus, model=model, model_note=note)
        except ValueError as error:
            print(f"MUGIURUSHI_AI_TOKEN: {error}", file=sys.stderr)
            return 2
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
