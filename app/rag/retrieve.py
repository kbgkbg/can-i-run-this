"""근거 검색 (F2006, F2014).

1. 선택한 모델의 model_meta 청크는 검색 없이 항상 [근거 1]로 넣는다.
2. model_docs에서 model_id 필터로 범위를 좁혀 질문과의 유사도로 top_k개 검색한다.
3. 개념 문서(위키백과)는 모델 필터 없이 따로 concept_top_k개 검색해 합친다.
   다른 모델 질문 때 모아 둔 개념 문서도 쓸 수 있게 하기 위해서다.
4. 청크 ID와 본문 해시로 중복을 제거하고 번호와 출처 유형 라벨을 붙인다.
검색 방식(similarity | mmr)과 수치는 config.yaml retrieval 절에서 정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document as LCDocument

from app.collect.sources.base import SOURCE_LABELS
from app.config import get_config
from app.index.indexer import content_hash
from app.index.store import DOCS_COLLECTION, META_COLLECTION, get_store


@dataclass
class Evidence:
    n: int                 # 근거 번호 (프롬프트와 화면에서 [근거 n])
    chunk_id: str
    text: str
    source_type: str
    title: str
    url: str
    page: int | None = None
    score: float | None = None  # 코사인 거리 (작을수록 가깝다). 메타데이터는 검색하지 않으므로 None

    @property
    def label(self) -> str:
        base = SOURCE_LABELS.get(self.source_type, self.source_type)
        return f"{base} p.{self.page}" if self.page else base

    @property
    def tag(self) -> str:
        return f"[근거 {self.n}] ({self.label})"


def _meta_evidence(model_id: str) -> Evidence | None:
    got = get_store(META_COLLECTION).get(ids=[model_id], include=["documents", "metadatas"])
    if not got["ids"]:
        return None
    meta = got["metadatas"][0]
    return Evidence(0, model_id, got["documents"][0], "model_meta", meta["doc_title"], meta["source_url"])


def _search(question: str, where: dict[str, Any], k: int, cfg: dict[str, Any]) -> list[tuple[LCDocument, float | None]]:
    store = get_store(DOCS_COLLECTION)
    if cfg["search_type"] == "mmr":
        # MMR: fetch_k개를 넉넉히 가져와 질문 관련성과 서로 다른 정도를 함께 따져 k개를 고른다 (F2014)
        docs = store.max_marginal_relevance_search(
            question, k=k, fetch_k=cfg["mmr_fetch_k"], lambda_mult=cfg["mmr_lambda"], filter=where
        )
        return [(d, None) for d in docs]
    return store.similarity_search_with_score(question, k=k, filter=where)


def retrieve(model_id: str, question: str) -> list[Evidence]:
    cfg = get_config()["retrieval"]
    evidence: list[Evidence] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    meta = _meta_evidence(model_id)
    if meta:
        evidence.append(meta)
        seen_ids.add(meta.chunk_id)

    hits = _search(question, {"model_id": model_id}, cfg["top_k"], cfg)
    if cfg["concept_top_k"] > 0:
        hits += _search(question, {"source_type": "wikipedia"}, cfg["concept_top_k"], cfg)

    for doc, score in hits:
        m = doc.metadata
        chunk_id = doc.id or f"{m.get('model_id')}:{m.get('doc_id')}:?"
        h = m.get("content_hash") or content_hash(doc.page_content)
        # 같은 청크(같은 ID) 또는 다른 모델로 수집돼 ID만 다른 같은 본문(같은 해시)은 한 번만 넣는다
        if chunk_id in seen_ids or h in seen_hashes:
            continue
        seen_ids.add(chunk_id)
        seen_hashes.add(h)
        evidence.append(Evidence(0, chunk_id, doc.page_content, m["source_type"], m["doc_title"],
                                 m["source_url"], m.get("page"), score))

    for i, e in enumerate(evidence, 1):
        e.n = i
    return evidence
