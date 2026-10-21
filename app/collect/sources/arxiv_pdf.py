"""arXiv 기술 보고서 PDF (F2016 여러 문서 출처 결합). 원문 출처 = PDF URL + 쪽 번호.

후보: 선택 모델의 HF 태그에 있는 arxiv:<ID> (모델 제작사가 모델 카드에 연결한 논문).
  검색어로 arXiv 전체를 찾으면 무관한 논문이 너무 많아, 제작사가 직접 연결한 논문만 후보로 쓴다.
  후보 요약은 arXiv API의 제목·초록으로 만들고, 프롬프트 2가 다른 후보와 함께 선별한다.
본문: PDF를 받아 pypdf로 쪽마다 텍스트를 뽑는다. 청킹은 쪽 단위로 한 뒤 크기 분할한다 (indexer.split_document).
호출 규칙: arXiv API 이용 규칙에 따라 같은 호스트 호출 간격 3초 이상 (config.yaml http.min_interval_s).
이용 조건: 논문마다 라이선스가 다르다 (arXiv 초록 페이지에 표시). 근거로는 일부 쪽을 인용하고 원문 링크를 함께 보인다.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from pypdf import PdfReader

from app.collect import http
from app.collect.sources.base import Candidate, Document
from app.config import get_config

SOURCE_TYPE = "arxiv_pdf"
API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
LICENSE_NOTE = "논문별 라이선스 (arXiv 초록 페이지 참고)"


def arxiv_ids(model: dict[str, Any]) -> list[str]:
    return [t.split(":", 1)[1] for t in model.get("tags", []) if t.startswith("arxiv:")]


def candidates_for(model: dict[str, Any]) -> list[Candidate]:
    """모델 태그의 arXiv ID들을 한 번의 API 호출로 조회해 후보(제목 + 초록 앞부분)를 만든다."""
    ids = arxiv_ids(model)
    if not ids:
        return []
    xml = http.get(API, params={"id_list": ",".join(ids), "max_results": len(ids)}).text
    out = []
    for entry in ET.fromstring(xml).findall("a:entry", ATOM):
        abs_url = entry.findtext("a:id", default="", namespaces=ATOM).strip()
        m = re.search(r"abs/([^v\s]+)", abs_url)
        if not m:
            continue
        aid = m.group(1)
        title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=ATOM)).strip()
        summary = re.sub(r"\s+", " ", entry.findtext("a:summary", default="", namespaces=ATOM)).strip()
        out.append(Candidate(
            doc_id=f"{SOURCE_TYPE}:{aid}",
            source_type=SOURCE_TYPE,
            title=f"{title} (arXiv:{aid})",
            summary=f"{model['id']} 모델 카드가 연결한 논문. 초록: {summary[:300]}",
            url=f"https://arxiv.org/pdf/{aid}",
            query=model["id"],
            extra={"arxiv_id": aid},
        ))
    return out


def fetch(candidate: Candidate) -> Document | None:
    max_pages = get_config()["collect"]["max_pdf_pages"]
    pdf = http.get(candidate.url, timeout=60).content
    reader = PdfReader(io.BytesIO(pdf))
    pages = []
    for i, page in enumerate(reader.pages[:max_pages], 1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    if not pages:
        return None
    return Document(
        doc_id=candidate.doc_id,
        source_type=SOURCE_TYPE,
        title=candidate.title,
        source_url=candidate.url,
        text="\n\n".join(t for _, t in pages),
        query=candidate.query,
        collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        license=LICENSE_NOTE,
        pages=pages,
    )
