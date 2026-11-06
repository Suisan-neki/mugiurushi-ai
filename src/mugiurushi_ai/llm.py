"""モデルの呼び出し。JSONスキーマで形を固定した応答だけを受け取る。

試験では FakeModel を使い、実際のAPIを呼ぶ試験は分ける。
送ってよいのは、公開文書の断片と、架空のデータ・開発者が書いた説明だけ。実患者の情報・生のログ・認証情報は
送らない（docs/design.md「送るデータ」）。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .config import ModelConfig


class ModelError(RuntimeError):
    pass


class Model(Protocol):
    name: str

    def complete(
        self, *, instructions: str, payload: dict[str, Any], schema: dict[str, Any], name: str
    ) -> tuple[dict, dict[str, Any]]:
        """instructions を指示、payload を資料として渡し、schema に合うJSONと、呼び出しの記録を返す。

        記録には、指定したモデル名と、APIが返したモデル名・応答のID・使ったトークン数を入れる。
        """
        ...


class OpenAIModel:
    def __init__(self, config: ModelConfig):
        from openai import OpenAI

        self.name = config.model
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=180, max_retries=1)

    def complete(
        self, *, instructions: str, payload: dict[str, Any], schema: dict[str, Any], name: str
    ) -> tuple[dict, dict[str, Any]]:
        try:
            response = self._client.responses.create(
                model=self.name,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                # Responses の応答を OpenAI 側に保存させない（後から応答IDで取り出せる保存の設定）。
                # これは不正利用の監視のためのログなど、ほかの保存を止めるものではない
                # （https://developers.openai.com/api/docs/guides/your-data）。送る内容は架空のデータに限る。
                store=False,
            )
        except Exception as error:  # SDKの例外の種類を利用者へそのまま見せない
            raise ModelError(f"モデルの呼び出しに失敗しました: {type(error).__name__}") from error
        try:
            data = json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ModelError("モデルの応答がJSONではありません") from error
        usage = getattr(response, "usage", None)
        return data, {
            "requested_model": self.name,
            "response_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }


class FakeModel:
    """試験用。あらかじめ与えた応答を順に返し、受け取った指示と資料を記録する。"""

    def __init__(self, *responses: dict, name: str = "fake"):
        self.name = name
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, *, instructions: str, payload: dict[str, Any], schema: dict[str, Any], name: str
    ) -> tuple[dict, dict[str, Any]]:
        self.calls.append({"instructions": instructions, "payload": payload, "schema": schema, "name": name})
        if not self._responses:
            raise ModelError("試験用の応答がありません")
        meta = {
            "requested_model": self.name,
            "response_model": self.name,
            "response_id": f"fake-{len(self.calls)}",
        }
        return self._responses.pop(0), meta


def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """OpenAIの strict なJSONスキーマ用。すべて必須・追加の項目なし。"""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
