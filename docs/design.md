# 設計

## 目的と範囲

発表資料の「AIエージェント」（医療情報を扱うための要件の過不足を確認し、要望とあわせて検証案を整理する）と、「① AIエージェントによる検証案の整理」「③ 確認結果の提示、残っている課題の認識」を担う。試験の実行・観測・判定は麦漆が担い、このリポジトリでは行わない。

本人の要望（2026年11月5日）：①さまざまな医療情報系のガイドラインや品質保証の規格に則っているかをスキャンできるAI、②対話によってテスト範囲を決め、検証とテスト結果を提示できるAI。対象文書は「国内の公開ガイドライン」「医療機器ソフトウェアの規制」「セキュリティの公開基準」から始める（本人の選択）。

## 責務の分担

| 処理 | 担当 | 確かなこと |
| --- | --- | --- |
| 文書の取り込み | このリポジトリ | 取得先・取得日・SHA-256・ページ。版の名称は人が記入 |
| 断片の検索 | このリポジトリ（決定的） | 文字の2-gramと語の重なり。外部の呼び出しなし |
| 照合の候補 | AI（OpenAI） | 引用はそのリクエストで渡した断片だけ（サーバーが検証）。内容の妥当性は人が確認 |
| 試験範囲の下書き | AI | 観測の種類は麦漆の一覧だけ（サーバーが検証）。判定条件への対応づけは空 |
| 試験の実行・判定・部分ごとの状態 | 麦漆 | 実行記録と照合した結果 |
| 結果の説明 | AI | 状態と判定は麦漆の値を写す。AIは文だけ |

## HTTP API

認証は共有トークン（`X-Mugiurushi-AI-Token`、16文字以上）。既定は `127.0.0.1` で待ち受ける。入力は1MiBまで。モデルが未設定なら照合・対話は503。AIの応答がサーバーの検証を通らなければ502で、部分的な結果を返さない。

| 経路 | 入力 | 出力 |
| --- | --- | --- |
| `GET /v1/health` | — | モデルの設定状況、取り込み済みの文書 |
| `GET /v1/sources` | — | 文書の一覧と取り込みの記録 |
| `POST /v1/scan` | `system.summary`、`system.excerpts[{ref,text}]`、`source_ids`、`max_passages` | `findings[{status:"candidate", requirement, relevance, citations[{passage_id, source, document, version, page, origin, sha256, fetched_at, excerpt}], system_refs, verification{method, method_label, support, support_label, executable_check, how}, unknowns}]`、`not_covered`、`passages`、`limits` |
| `POST /v1/dialogue/turn` | `system`、`concern`、`messages`、`current_aspects` | `reply`、`questions`、`aspects[{aspect{id,text,origin,observation,trial_checks:[]}, support, rationale, unknowns}]`、`out_of_scope`、`status:"draft"` |
| `POST /v1/results/explain` | 麦漆の `check_text`・`verdict`・`verification_passed`・`coverage` | 麦漆の判定と状態をそのまま、部分ごとの `explanation`、`next_steps` |

`aspects[].aspect` は麦漆の `set_aspects` にそのまま渡せる形にした。観測の種類の一覧は麦漆の `src/coverage.rs`（main `e3a4201`）から写した（`src/mugiurushi_ai/kinds.py`）。

`sources` は、指定した出典（指定がなければ一覧のすべて）ごとの状態をサーバーが返す：`not_in_catalog`（一覧にない）・`not_ingested`（取り込まれていない）・`no_matching_passages`（取り込み済みだが説明と重なる断片が無く、モデルに渡していない）・`used`（断片をモデルに渡した）。モデルの `not_covered` には任せない。

取り込みは文書ごとに、取得記録（取得先・取得日・SHA-256）と断片を一つのファイル（`corpus/data/<文書ID>/snapshot.json`）にまとめ、一時ファイルから置き換える。照合は一度に読んだスナップショットから断片と取得記録を取るため、照合と同時に新版を取り込んでも、旧版の本文に新版のSHA-256を付けない。取り込みに失敗した場合は、以前の有効な版が残る。

`verification.support` は、その確認の方法を麦漆がどこまで扱えるか（`judged` 判定できる・`recorded_only` 記録のみ・`inference` 推定・`unsupported` 麦漆では観測できない・`human` 人による確認）。`executable_check` は `judged` のときだけ真で、それ以外の候補は試験が通っても未確認のまま残る。各応答の `model_call` には、指定したモデル名・APIが返したモデル名・応答のID・トークン数を記録する。

## 送るデータ

- OpenAIへ送るのは、公開文書の断片、開発者が書いたシステムの説明・コードの抜粋、麦漆の結果のうち確認項目の文と部分ごとの状態・説明だけ。
- 接続の試験とデモは架空のデータだけで行う。実患者の情報・生のログ・認証情報は送らない。麦漆から呼ぶときも、実行記録の生ログや画面写真、試験用の認証情報は送らない。
- 応答は `store=False` で送る。これは Responses の応答を OpenAI 側に保存させない設定で、不正利用の監視のためのログなど、ほかの保存を止めるものではない（[OpenAIのデータの扱い](https://developers.openai.com/api/docs/guides/your-data)）。
- モデルの接続先・APIキー・共有トークンは運営者がサーバー側で設定する。麦漆の画面から入力させない。

## 麦漆への組み込み（次の作業）

麦漆の画面・APIから、運営者が設定したこのサービスを呼ぶ。

1. 対話の画面で `/v1/dialogue/turn` を呼び、案を「確かめたい部分」の下書きとして登録する（現在の対話の整理案の引き継ぎと同じく、人が観測の種類と判定条件を選ぶまで未確認）。
2. 結果画面で `/v1/results/explain` を呼び、麦漆の状態の横に説明を示す。
3. 構成・コードの読取結果から `/v1/scan` を呼び、照合の候補を示す。候補から試験を作る場合は 1 と同じ経路にする。

## 未実装・未確認

- 実際のOpenAI APIでの動作（APIキーの登録後に `tests/live` で確かめる）。回答の品質の評価。
- 埋め込みによる検索。断片の検索は語の重なりだけ。
- 提供事業者向けガイドライン（経済産業省のPDF）の取り込み。この開発環境から取得できない。
- 規格の本文（ISO・IECなど）は扱わない。条項番号と、本人が書いた要約で扱うかは未決。
- 麦漆への組み込み。
