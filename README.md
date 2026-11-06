# mugiurushi-ai

[麦漆（mugiurushi）](https://github.com/Suisan-neki/mugiurushi)に組み込むAIエージェント。OpenAI APIを使う。

1. **ガイドラインとの照合**：システムの説明（構成・コードの抜粋）を、取り込んだ医療情報系のガイドライン・医療機器ソフトウェアの規制・セキュリティの公開基準と照らし、該当しうる要求・確かめ方・未確認の点を、出典（文書・ページ・取得した文書のSHA-256）つきで返す。
2. **対話による試験範囲の決定と結果の説明**：開発者の不安から、麦漆の「確かめる部分」の下書きを作る。麦漆の試験結果を受け取り、麦漆が観測したこと・未確認のことを分けて説明する。

試験の実行と判定は麦漆が行う。このリポジトリのAIは、候補・下書き・説明を作るだけで、合否や観測の状態を決めない。照合の結果は適合・準拠・安全性の判断ではない。設計と麦漆との接続は [docs/design.md](docs/design.md)、開発の決まりは [AGENTS.md](AGENTS.md)。

## 照合に使う文書

一覧は [corpus/catalog.json](corpus/catalog.json)。公開元の掲載ページで題名と取得先を確認した日付を記録し、版の名称は取り込んだ文書を人が確認して記入する。

| 文書 | 公開元 | この開発環境での取り込み（2026-11-05） |
| --- | --- | --- |
| 医療情報システムの安全管理に関するガイドライン（概説編・経営管理編・企画管理編・システム運用編・Q&A） | 厚生労働省 | 取り込めた。表紙は「第 7.0 版」 |
| 医療情報を取り扱う情報システム・サービスの提供事業者における安全管理ガイドライン | 総務省・経済産業省 | 取得先（経済産業省）がHTTP 403。`--file` で取り込む |
| 医療・介護関係事業者における個人情報の適切な取扱いのためのガイダンス | 個人情報保護委員会・厚生労働省 | 取り込めた。表紙は「令和８年４月一部改正」 |
| プログラムの医療機器該当性に関するガイドライン | 厚生労働省 | 取り込めた。冒頭は令和5年3月31日付の通知（題名は人が確認する） |
| OWASP ASVS 5.0 | OWASP Foundation（CC BY-SA 4.0） | 取り込めた |
| NIST SP 800-53 Rev. 5 | NIST | 取り込めた |

取り込んだ本文は `corpus/data/` に保存し、リポジトリには入れない。有料の規格（ISO・IECなど）の本文は扱わない。

## 使い方

```sh
uv sync
uv run mugiurushi-ai sources
uv run mugiurushi-ai ingest --all
uv run mugiurushi-ai ingest provider-security-main --file ./01gl_20250328.pdf
```

サーバーは運営者が環境変数で設定して起動する。モデル名は推測で補わず、未設定なら照合・対話は「未設定」と返す。

```sh
MUGIURUSHI_AI_TOKEN=<16文字以上の共有トークン> \
MUGIURUSHI_AI_MODEL=<OpenAIのモデル名> \
OPENAI_API_KEY=<キー> \
uv run mugiurushi-ai serve
```

プロキシが `api.openai.com` への認証を付ける環境（Claude Code の Network secret など）では、`OPENAI_API_KEY` の代わりに `MUGIURUSHI_AI_AUTH=proxy` を指定する。

送るのは公開文書の断片と、架空のデータ・開発者が書いた説明だけで、実患者の情報・生のログ・認証情報は送らない。`store=False` は Responses の応答を保存させない設定で、不正利用の監視のためのログなどほかの保存は止めない（[docs/design.md](docs/design.md)「送るデータ」）。

## 確認

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

試験は偽のモデルと架空の文書で行う。実際のOpenAI APIを使う試験（`tests/live`）は `MUGIURUSHI_AI_LIVE=1` のときだけ実行する。
