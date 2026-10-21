"""Hugging Face 모델 카드 (주 출처). 원문 출처 = 모델 페이지 URL."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from app.collect import http
from app.collect.sources.base import Candidate, Document
from app.config import env

SOURCE_TYPE = "hf_card"
HF = "https://huggingface.co"
LICENSE_NOTE = "모델 카드 원문의 라이선스를 따름 (모델별 license 필드 참고)"

_SUMMARY_TAG_SKIP = ("arxiv:", "base_model:", "region:", "deploy:", "dataset:")


def _headers() -> dict[str, str]:
    token = env("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def doc_id_for(model_id: str) -> str:
    return f"{SOURCE_TYPE}:{model_id}"


def card_candidate(model_id: str, query: str = "") -> Candidate:
    """선택한 모델 자체의 모델 카드 후보 (검색 없이 만든다)."""
    return Candidate(
        doc_id=doc_id_for(model_id),
        source_type=SOURCE_TYPE,
        title=f"{model_id} 모델 카드",
        summary="선택한 모델의 공식 모델 카드",
        url=f"{HF}/{model_id}",
        query=query or model_id,
        extra={"model_id": model_id},
    )


def search(query: str, limit: int) -> list[Candidate]:
    """HF 모델 저장소 이름 검색. 요약은 목록 응답(태그·다운로드)으로만 만든다."""
    params = {"search": query, "limit": limit, "sort": "downloads", "direction": "-1"}
    results = http.get_json(f"{HF}/api/models", params=params, headers=_headers())
    candidates = []
    for m in results:
        tags = [t for t in m.get("tags", []) if not t.startswith(_SUMMARY_TAG_SKIP)][:8]
        summary = f"HF 모델 저장소, 다운로드 {m.get('downloads', 0):,}회, 태그: {', '.join(tags)}"
        candidates.append(Candidate(
            doc_id=doc_id_for(m["id"]),
            source_type=SOURCE_TYPE,
            title=f"{m['id']} 모델 카드",
            summary=summary,
            url=f"{HF}/{m['id']}",
            query=query,
            extra={"model_id": m["id"]},
        ))
    return candidates


def clean_card(markdown: str) -> str:
    """YAML 머리말, HTML 태그, 이미지/배지를 걷어내 본문만 남긴다."""
    text = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)            # 마크다운 이미지
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)                          # HTML 태그
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch(candidate: Candidate) -> Document | None:
    model_id = candidate.extra["model_id"]
    try:
        raw = http.get(f"{HF}/{model_id}/raw/main/README.md", headers=_headers()).text
    except http.ExternalAPIError:
        return None  # 모델 카드가 없는 저장소
    text = clean_card(raw)
    if not text:
        return None
    return Document(
        doc_id=candidate.doc_id,
        source_type=SOURCE_TYPE,
        title=candidate.title,
        source_url=candidate.url,
        text=text,
        query=candidate.query,
        collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        license=LICENSE_NOTE,
    )
