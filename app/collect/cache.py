"""수집 캐시 (NF2002).

- data/cache/collected.json: 모델별로 이미 수집한 검색어(정규화)와 색인한 문서 목록
- data/cache/docs/*.json: 문서 원문. 다른 모델이 같은 문서를 고르면 본문을 다시 받지 않고 재사용하고,
  청크 크기를 바꿔 재색인할 때도 이 원문을 쓴다.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from app.collect.sources.base import Document
from app.config import path


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _index_file():
    return path("cache_dir") / "collected.json"


def _doc_file(doc_id: str):
    safe = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:20]
    return path("cache_dir") / "docs" / f"{safe}.json"


def load() -> dict[str, Any]:
    f = _index_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save(data: dict[str, Any]) -> None:
    f = _index_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)


def model_entry(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    return data.setdefault(model_id, {"queries": {}, "docs": {}})


def record(data: dict[str, Any], model_id: str, queries: list[str], docs: list[Document]) -> None:
    entry = model_entry(data, model_id)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for q in queries:
        entry["queries"][normalize_query(q)] = {
            "collected_at": now,
            "doc_ids": [d.doc_id for d in docs if normalize_query(d.query) == normalize_query(q)],
        }
    for d in docs:
        entry["docs"][d.doc_id] = {"title": d.title, "source_type": d.source_type, "source_url": d.source_url}


def indexed_docs(model_id: str) -> dict[str, dict[str, str]]:
    """모델에 색인된 문서 {doc_id: {title, source_type, source_url}} (사이드바 표시용)."""
    return load().get(model_id, {}).get("docs", {})


def load_doc(doc_id: str) -> Document | None:
    f = _doc_file(doc_id)
    return Document.from_dict(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else None


def save_doc(doc: Document) -> None:
    f = _doc_file(doc.doc_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")


def all_docs() -> list[Document]:
    d = path("cache_dir") / "docs"
    return [Document.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(d.glob("*.json"))] if d.exists() else []
