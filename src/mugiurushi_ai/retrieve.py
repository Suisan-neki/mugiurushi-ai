"""照合に使う断片の検索。

日本語の文に区切りがないため、文字の2-gramと英数字の語の重なりで順位を付ける。
決定的で外部の呼び出しがない。埋め込みによる検索に差し替える場合も、この関数の形を保つ。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Chunk

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def terms(text: str) -> Counter[str]:
    text = text.lower()
    words = Counter(_WORD.findall(text))
    cjk = re.sub(r"[A-Za-z0-9\s\W_]+", " ", text)
    grams = Counter(part[i : i + 2] for part in cjk.split() for i in range(len(part) - 1) if len(part) >= 2)
    return words + grams


def search(chunks: list[Chunk], query: str, limit: int = 12) -> list[Chunk]:
    """query との重なりが大きい断片を返す。重なりのない断片は返さない。"""
    wanted = terms(query)
    if not wanted:
        return []
    documents = [terms(c.text) for c in chunks]
    frequency: Counter[str] = Counter()
    for d in documents:
        frequency.update(d.keys())
    total = len(chunks)
    scored = []
    for chunk, d in zip(chunks, documents, strict=True):
        score = sum(
            min(count, d[t]) * math.log(1 + total / frequency[t]) for t, count in wanted.items() if t in d
        )
        if score > 0:
            scored.append((score / math.sqrt(1 + sum(d.values())), chunk.id, chunk))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored[:limit]]


def select(matched: list[Chunk], limit: int) -> list[Chunk]:
    """順位付けした断片から、モデルに渡す断片を選ぶ。

    まず説明と重なる断片がある出典ごとに最上位の1件を（出典の最上位の順に）取り、残りの枠を全体の順位で埋める。
    全体の順位だけで選ぶと、断片の多い出典や文の短い出典が上位を占め、重なる断片が多い出典でも1件も渡らない
    ことがある。出典の数が枠より多い場合は、最上位の断片の順位が低い出典が外れる。返す順は全体の順位のまま。
    """
    chosen: set[str] = set()
    covered: set[str] = set()
    for chunk in matched:
        if len(chosen) >= limit:
            break
        if chunk.source_id not in covered:
            covered.add(chunk.source_id)
            chosen.add(chunk.id)
    for chunk in matched:
        if len(chosen) >= limit:
            break
        chosen.add(chunk.id)
    return [c for c in matched if c.id in chosen]
