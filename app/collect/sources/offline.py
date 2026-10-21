"""오프라인 문서 묶음 (외부 API를 쓸 수 없는 환경용, F2004).

data/offline_docs/*.json 파일 하나가 Document 1건(Document.to_dict() 형식)이다.
`python scripts/export_offline_docs.py`로 이미 수집한 문서를 묶음으로 내보낼 수 있다.
search()는 제목·본문 키워드 일치로 후보를 고르고, fetch()는 파일 내용을 그대로 돌려준다.
"""
from __future__ import annotations

import json
from functools import lru_cache

from app.collect.sources.base import Candidate, Document
from app.config import path


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Document]:
    docs = {}
    for f in sorted(path("offline_docs_dir").glob("*.json")):
        doc = Document.from_dict(json.loads(f.read_text(encoding="utf-8")))
        docs[doc.doc_id] = doc
    return docs


def _to_candidate(doc: Document, query: str) -> Candidate:
    extra = {"model_id": doc.doc_id.split(":", 1)[1]} if doc.source_type == "hf_card" else {}
    return Candidate(doc.doc_id, doc.source_type, doc.title, doc.text[:200], doc.source_url, query, extra)


def search(query: str, limit: int) -> list[Candidate]:
    terms = [t.lower() for t in query.split() if t]
    scored = []
    for doc in _bundle().values():
        hay = (doc.title + " " + doc.text[:5000]).lower()
        score = sum(hay.count(t) for t in terms)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [_to_candidate(doc, query) for _, doc in scored[:limit]]


def get(doc_id: str) -> Document | None:
    return _bundle().get(doc_id)


def fetch(candidate: Candidate) -> Document | None:
    return get(candidate.doc_id)
