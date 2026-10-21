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


# 문장 속 독립된 수치 (앞뒤가 영문자가 아닌 숫자). "Qwen3", "8B" 같은 이름 일부는 빼고 "0.6", "40,960", "36조"는 잡는다.
_NUM = re.compile(r"(?<![A-Za-z0-9.])\d+(?:[.,]\d+)*(?![A-Za-z\d])")


def _norm_numbers(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 40,960 -> 40960


def _has_number(text: str, num: str) -> bool:
    # "50"이 "50.0"이나 "150" 안에서 맞지 않게 앞뒤 숫자·소수부를 막는다
    return re.search(rf"(?<![\d.]){re.escape(num)}(?!\.?\d)", _norm_numbers(text)) is not None


def unverified_numbers(answer: str, evidence: list[Evidence], calc: str = "") -> list[str]:
    """[근거 n]을 붙인 줄의 수치가 그 근거 본문(계산값 줄이면 계산값)에 실제로 있는지 검사한다.

    근거를 인용하는 모양은 갖췄지만 값은 지어낸 경우(예: 근거에 없는 "Top-k 50")를 잡기 위한 장치다.
    반환: 확인되지 않은 수치가 있는 줄의 요약 목록 (화면 경고용). 문맥까지 보는 검증은 아니다.
    """
    by_n = {e.n: e for e in evidence}
    problems = []
    for line in answer.splitlines():
        nums_cited = [int(n) for g in _CITE.findall(line) for n in re.findall(r"\d+", g)]
        if not nums_cited and "계산값" not in line:
            continue
        body = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", _CITE.sub(" ", line))  # 줄 앞 목록 번호("3.")는 수치가 아니다
        sources = " ".join(by_n[n].text for n in nums_cited if n in by_n)
        if "계산값" in line:
            sources += " " + calc
        missing = [n for n in _NUM.findall(_norm_numbers(body)) if not _has_number(sources, n)]
        if missing:
            problems.append(f"{line.strip()[:60]} (근거에 없는 수치: {', '.join(missing)})")
    return problems
