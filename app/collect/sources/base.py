"""출처 어댑터 공통 자료형.

어댑터는 두 단계로 나눈다.
- search(): 후보의 제목 + 짧은 요약만 가져온다 (본문 X). 프롬프트 2가 이것만 보고 고른다.
- fetch(): 선별된 후보의 본문과 원문 출처를 가져온다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# 출처 유형 -> 화면 라벨 (근거 표시에 사용)
SOURCE_LABELS = {
    "model_meta": "모델 메타데이터",
    "hf_card": "HF 모델 카드",
    "hf_license": "라이선스 원문",
    "wikipedia": "위키백과",
    "arxiv_pdf": "arXiv PDF",
}


@dataclass
class Candidate:
    doc_id: str          # 출처 안에서 고유한 문서 ID (예: hf_card:Qwen/Qwen3-8B, wikipedia:ko:12345)
    source_type: str     # hf_card | hf_license | wikipedia | arxiv_pdf
    title: str
    summary: str
    url: str
    query: str = ""      # 이 후보를 찾은 검색어
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    doc_id: str
    source_type: str
    title: str
    source_url: str
    text: str
    query: str
    collected_at: str
    license: str = ""
    # PDF처럼 쪽 번호가 있는 출처: [(쪽 번호, 그 쪽 텍스트), ...]. 없으면 text만 쓴다.
    pages: list[tuple[int, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        d = dict(d)
        if d.get("pages") is not None:
            d["pages"] = [tuple(p) for p in d["pages"]]
        return cls(**d)
