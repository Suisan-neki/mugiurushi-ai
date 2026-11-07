"""麦漆から呼ぶHTTP API。

認証は運営者が設定する共有トークン（ヘッダー X-Mugiurushi-AI-Token）。モデルが未設定なら、
照合・対話は 503 と未設定の理由を返し、固定の文で代わりに答えない。
"""

from __future__ import annotations

import hmac
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from . import corpus as corpus_module
from .dialogue import DialogueError, ExplainRequest, TurnRequest, explain, turn
from .flow import AskRequest, FlowError, PlanRequest, ask, plan
from .llm import Model, ModelError
from .scan import ScanError, ScanRequest, scan

MAX_BODY = 1024 * 1024


class BodyLimit:
    """入力の大きさを、実際に受け取ったバイト数で制限する（Content-Length の有無によらない）。

    本文をすべて受け取ってから処理へ渡すため、上限を超えた入力では処理（モデルの呼び出し）に進まない。
    """

    def __init__(self, app, limit: int = MAX_BODY):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.limit):
            return await self._reject(scope, receive, send)
        body = b""
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if len(body) > self.limit:
                return await self._reject(scope, receive, send)
            more = message.get("more_body", False)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(scope, receive, send):
        await JSONResponse({"error": "入力は1MiBまでです"}, status_code=413)(scope, receive, send)


def create_app(*, token: str, corpus: Path, model: Model | None, model_note: str = "") -> FastAPI:
    if len(token) < 16:
        raise ValueError("共有トークンは16文字以上にしてください")
    app = FastAPI(title="mugiurushi-ai", docs_url=None, redoc_url=None, openapi_url=None)

    app.add_middleware(BodyLimit)

    def authorized(x_mugiurushi_ai_token: str = Header(default="")) -> None:
        if not hmac.compare_digest(x_mugiurushi_ai_token.encode(), token.encode()):
            raise HTTPException(status_code=403, detail="トークンが一致しません")

    def require_model() -> Model:
        if model is None:
            raise HTTPException(status_code=503, detail=f"AIモデルが未設定です。{model_note}".strip())
        return model

    @app.get("/v1/health", dependencies=[Depends(authorized)])
    def health():
        records = corpus_module.records(corpus)
        return {
            "model": model.name if model else None,
            "model_status": "configured" if model else "not_configured",
            "ingested_documents": sorted(records),
        }

    @app.get("/v1/sources", dependencies=[Depends(authorized)])
    def sources():
        catalog = corpus_module.load_catalog(corpus)
        records = corpus_module.records(corpus)
        return {
            "note": catalog.note,
            "sources": [
                {
                    **source.model_dump(exclude={"documents"}),
                    "documents": [
                        {**d.model_dump(), "ingested": asdict(records[d.id]) if d.id in records else None}
                        for d in source.documents
                    ],
                }
                for source in catalog.sources
            ],
        }

    @app.post("/v1/scan", dependencies=[Depends(authorized)])
    def scan_endpoint(body: ScanRequest, model: Model = Depends(require_model)):
        catalog = corpus_module.load_catalog(corpus)
        try:
            # 断片と取得記録を一度に読んだ同じスナップショットから取る（取り込みと重なっても混ざらない）。
            return scan(body, corpus_module.load_snapshots(corpus), catalog, model)
        except (ScanError, ModelError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/v1/ask", dependencies=[Depends(authorized)])
    def ask_endpoint(body: AskRequest, model: Model = Depends(require_model)):
        try:
            return ask(body, corpus_module.load_snapshots(corpus), corpus_module.load_catalog(corpus), model)
        except (FlowError, ModelError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/v1/plan", dependencies=[Depends(authorized)])
    def plan_endpoint(body: PlanRequest, model: Model = Depends(require_model)):
        try:
            return plan(body, model)
        except (FlowError, ModelError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/v1/dialogue/turn", dependencies=[Depends(authorized)])
    def turn_endpoint(body: TurnRequest, model: Model = Depends(require_model)):
        try:
            return turn(body, model)
        except (DialogueError, ModelError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/v1/results/explain", dependencies=[Depends(authorized)])
    def explain_endpoint(body: ExplainRequest, model: Model = Depends(require_model)):
        try:
            return explain(body, model)
        except (DialogueError, ModelError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app
