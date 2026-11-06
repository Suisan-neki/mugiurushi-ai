"""麦漆の観測の種類（mugiurushi の src/coverage.rs の ObservationKind と support）。

AIの試験範囲の案は、この一覧の種類だけを使う。麦漆側でも登録時に検査する。
写した元: Suisan-neki/mugiurushi main（e3a4201）の src/coverage.rs。
"""

from __future__ import annotations

JUDGED = "judged"
RECORDED_ONLY = "recorded_only"
INFERENCE = "inference"
UNSUPPORTED = "unsupported"

OBSERVATION_KINDS: dict[str, str] = {
    "exit_code": JUDGED,
    "stdout": JUDGED,
    "stderr": JUDGED,
    "stored_file": JUDGED,
    "readback_output": JUDGED,
    "server_output": JUDGED,
    "container_log": JUDGED,
    "database_query": JUDGED,
    "browser_text": JUDGED,
    "browser_response": JUDGED,
    "browser_canvas": JUDGED,
    "server_http": JUDGED,
    "database_rows": JUDGED,
    "database_relation": JUDGED,
    "program_observation": JUDGED,
    "reported_observation": JUDGED,
    "traffic_summary": RECORDED_ONLY,
    "code_inspection": INFERENCE,
    "network_payload": UNSUPPORTED,
    "distributed_trace": UNSUPPORTED,
    "external_destination": UNSUPPORTED,
    "production_environment": UNSUPPORTED,
}

HUMAN = "human"

# 麦漆がその方法でどこまで扱えるか。executable は、麦漆の試験で観測して判定できる場合だけ真。
SUPPORT_LABELS = {
    JUDGED: "麦漆の試験で観測し、判定できる",
    RECORDED_ONLY: "麦漆の記録に残るが、判定には使えない",
    INFERENCE: "推定（実行の観測ではない）",
    UNSUPPORTED: "麦漆では観測できない",
    HUMAN: "人による確認（麦漆の判定ではない）",
}


def support_of(method: str) -> str:
    """確認の方法が、麦漆でどこまで扱えるか。人による方法は human。"""
    return OBSERVATION_KINDS.get(method, HUMAN)


# 照合（scan）で、麦漆の観測では確かめられない確認の方法。
HUMAN_METHODS = {
    "document_review": "文書・手順書の確認（人）",
    "interview": "担当者への確認（人）",
    "code_review": "コードの確認（人・AIの推定。実行の観測ではない）",
}
