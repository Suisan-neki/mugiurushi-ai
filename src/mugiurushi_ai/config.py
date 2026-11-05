"""運営者が起動時に与える設定。HTTPからは変更しない。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# プロキシが認証ヘッダーを付ける環境で使う、秘密でない固定文字列。実際のキーではない。
PROXY_PLACEHOLDER = "proxy-injected-placeholder"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key: str

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> ModelConfig | None:
        """`MUGIURUSHI_AI_MODEL` が無ければ未設定（None）。モデル名を推測で補わない。

        認証は `OPENAI_API_KEY`、またはプロキシが認証を付ける場合の `MUGIURUSHI_AI_AUTH=proxy`。
        """
        env = dict(os.environ) if env is None else env
        model = env.get("MUGIURUSHI_AI_MODEL", "").strip()
        if not model:
            return None
        base_url = env.get("MUGIURUSHI_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = env.get("OPENAI_API_KEY", "")
        if not key:
            if env.get("MUGIURUSHI_AI_AUTH") != "proxy":
                raise ConfigError(
                    "OPENAI_API_KEY を設定するか、プロキシが認証を付ける場合は MUGIURUSHI_AI_AUTH=proxy を指定してください"
                )
            key = PROXY_PLACEHOLDER
        return ModelConfig(model=model, base_url=base_url, api_key=key)


def corpus_dir(env: dict[str, str] | None = None) -> Path:
    env = dict(os.environ) if env is None else env
    return Path(env.get("MUGIURUSHI_AI_CORPUS", Path(__file__).resolve().parents[2] / "corpus"))
