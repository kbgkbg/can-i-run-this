"""위키백과 (개념 문서: 양자화, 트랜스포머 등). 한국어 우선, 결과가 없으면 영어.

MediaWiki API 사용. User-Agent(프로젝트명 + 연락처)는 http 모듈이 붙인다 (NF2006).
이용 조건: CC BY-SA 4.0 (https://ko.wikipedia.org/wiki/위키백과:저작권)
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from urllib.parse import quote

from app.collect import http
from app.collect.sources.base import Candidate, Document

SOURCE_TYPE = "wikipedia"
LICENSE_NOTE = "CC BY-SA 4.0 (위키백과)"


def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def search(query: str, limit: int, langs: list[str]) -> list[Candidate]:
    """langs 순서대로 검색해 결과가 나온 첫 언어의 후보를 반환한다."""
    for lang in langs:
        data = http.get_json(_api(lang), params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json", "formatversion": 2,
        })
        hits = data.get("query", {}).get("search", [])
        if hits:
            return [
                Candidate(
                    doc_id=f"{SOURCE_TYPE}:{lang}:{h['pageid']}",
                    source_type=SOURCE_TYPE,
                    title=f"{h['title']} ({lang}.wikipedia)",
                    summary=_strip_html(h.get("snippet", "")),
                    url=f"https://{lang}.wikipedia.org/wiki/{quote(h['title'].replace(' ', '_'))}",
                    query=query,
                    extra={"lang": lang, "pageid": h["pageid"]},
                )
                for h in hits
            ]
    return []


def fetch(candidate: Candidate) -> Document | None:
    lang, pageid = candidate.extra["lang"], candidate.extra["pageid"]
    data = http.get_json(_api(lang), params={
        "action": "query", "prop": "extracts", "explaintext": 1,
        "pageids": pageid, "format": "json", "formatversion": 2,
    })
    pages = data.get("query", {}).get("pages", [])
    text = (pages[0].get("extract") or "").strip() if pages else ""
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
