"""モデルの呼び出し。JSONスキーマで形を固定した応答だけを受け取る。

試験では FakeModel を使い、実際のAPIを呼ぶ試験は分ける。
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
    ) -> dict:
        """instructions を指示、payload を資料として渡し、schema に合うJSONを返す。"""
        ...


class OpenAIModel:
    def __init__(self, config: ModelConfig):
        from openai import OpenAI

        self.name = config.model
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=180, max_retries=1)

    def complete(
        self, *, instructions: str, payload: dict[str, Any], schema: dict[str, Any], name: str
    ) -> dict:
        try:
            response = self._client.responses.create(
                model=self.name,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                # 医療情報を含みうるため、OpenAI側に応答を保存させない。
                store=False,
            )
        except Exception as error:  # SDKの例外の種類を利用者へそのまま見せない
            raise ModelError(f"モデルの呼び出しに失敗しました: {type(error).__name__}") from error
        try:
            return json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ModelError("モデルの応答がJSONではありません") from error


class FakeModel:
    """試験用。あらかじめ与えた応答を順に返し、受け取った指示と資料を記録する。"""

    def __init__(self, *responses: dict, name: str = "fake"):
        self.name = name
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, *, instructions: str, payload: dict[str, Any], schema: dict[str, Any], name: str
    ) -> dict:
        self.calls.append({"instructions": instructions, "payload": payload, "schema": schema, "name": name})
        if not self._responses:
            raise ModelError("試験用の応答がありません")
        return self._responses.pop(0)


def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """OpenAIの strict なJSONスキーマ用。すべて必須・追加の項目なし。"""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
