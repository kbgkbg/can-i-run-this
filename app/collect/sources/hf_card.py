"""Hugging Face 모델 카드 (주 출처)와 라이선스 원문.

- 모델 카드: 저장소의 README.md. 원문 출처 = 모델 페이지 URL
- 라이선스 원문: 라이선스 질문일 때 저장소의 LICENSE 파일. 원문 출처 = 파일 URL
  라이선스 이름(apache-2.0 등)만으로 조건을 말하면 근거 없는 사전 지식이 섞여서, 원문을 근거로 넣는다.
"""
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


LICENSE_SOURCE_TYPE = "hf_license"
_DEFAULT_LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt")


def license_doc_id(model_id: str) -> str:
    return f"{LICENSE_SOURCE_TYPE}:{model_id}"


def _license_files(model_id: str, license_link: str | None) -> list[str]:
    """시도할 LICENSE 파일 경로. license_link가 같은 저장소의 파일이면 그것부터."""
    files: list[str] = []
    if license_link:
        m = re.match(rf"https://huggingface\.co/{re.escape(model_id)}/(?:blob|resolve|raw)/main/(.+)$", license_link)
        if m:
            files.append(m.group(1))
        elif not license_link.startswith("http"):
            files.append(license_link.lstrip("./"))
    return list(dict.fromkeys(files + list(_DEFAULT_LICENSE_FILES)))


def license_candidate(model_id: str, license_link: str | None) -> Candidate:
    return Candidate(
        doc_id=license_doc_id(model_id),
        source_type=LICENSE_SOURCE_TYPE,
        title=f"{model_id} 라이선스 원문",
        summary="선택한 모델 저장소의 LICENSE 파일",
        url=f"{HF}/{model_id}",
        query="라이선스",
        extra={"model_id": model_id, "files": _license_files(model_id, license_link)},
    )


def fetch_license(candidate: Candidate) -> Document | None:
    """LICENSE 파일을 순서대로 시도한다. 하나도 없으면 None, gated(401)면 ExternalAPIError."""
    model_id = candidate.extra["model_id"]
    for f in candidate.extra["files"]:
        try:
            text = http.get(f"{HF}/{model_id}/raw/main/{f}", headers=_headers()).text.strip()
        except http.ExternalAPIError as e:
            if "401" in str(e):
                raise  # gated: 토큰 없이는 원문을 받을 수 없다
            continue
        if text:
            return Document(
                doc_id=candidate.doc_id,
                source_type=LICENSE_SOURCE_TYPE,
                title=candidate.title,
                source_url=f"{HF}/{model_id}/blob/main/{f}",
                text=text,
                query=candidate.query,
                collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                license="라이선스 원문 자체",
            )
    return None
