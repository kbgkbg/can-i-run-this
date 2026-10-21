"""답변 본문의 [근거 n] 표기를 파싱해 실제로 인용된 근거만 돌려준다 (F2008)."""
from __future__ import annotations

import re

from app.rag.retrieve import Evidence

NO_CITATION_NOTICE = "이 답변에는 인용된 근거가 없습니다."

# [근거 3], [근거 1, 3], [근거 1·3], [근거1] 모두 허용
_CITE = re.compile(r"\[근거\s*([\d\s,·]+)\]")


def cited_numbers(answer: str) -> list[int]:
    """본문에 나온 순서대로, 중복 없이 근거 번호를 반환한다."""
    seen: list[int] = []
    for group in _CITE.findall(answer):
        for n in re.findall(r"\d+", group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def cited_evidence(answer: str, evidence: list[Evidence]) -> list[Evidence]:
    """실제로 인용된 근거만 번호 순서로. 근거 목록에 없는 번호(환각 인용)는 버린다."""
    by_n = {e.n: e for e in evidence}
    return sorted((by_n[n] for n in cited_numbers(answer) if n in by_n), key=lambda e: e.n)


def invalid_citations(answer: str, evidence: list[Evidence]) -> list[int]:
    """근거 목록에 없는 번호를 인용했는지 확인 (검증용)."""
    valid = {e.n for e in evidence}
    return [n for n in cited_numbers(answer) if n not in valid]
