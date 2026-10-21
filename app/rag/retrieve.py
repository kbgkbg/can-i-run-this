"""근거 검색 (F2006, F2014).

1. 선택한 모델의 model_meta 청크는 검색 없이 항상 [근거 1]로 넣는다.
2. model_docs에서 model_id 필터로 범위를 좁혀 질문과의 유사도로 검색한다.
3. 개념 문서(위키백과)는 모델 필터 없이 따로 검색해 합친다. 다른 모델 질문 때 모아 둔 개념 문서도 쓰기 위해서다.
   - 프롬프트 1이 개념 검색어를 뽑았을 때만, 그 검색어로 검색한다.
     질문 전체 문장으로 찾으면 개념 질문이 아니어도 가장 가까운 위키 청크가 늘 끼어들었다.
4. 중복 제거: 같은 청크 ID, 같은 본문 해시, 그리고 거의 같은 본문(단어 3-gram 자카드 유사도).
   공식 카드를 그대로 옮긴 GGUF·AWQ 카드처럼 헤더 한 줄만 다른 청크가 근거 자리를 차지하지 않게 한다.
   빠질 몫까지 top_k × overfetch개를 먼저 가져와 중복을 걸러낸 뒤 top_k개를 채운다.
검색 방식(similarity | mmr)과 수치는 config.yaml retrieval 절에서 정한다.
"""
from __future__ import annotations

import re
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
    score: float | None = None  # 코사인 거리 (작을수록 가깝다). 메타데이터와 MMR 결과는 None

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


def _search(query: str, where: dict[str, Any], k: int, cfg: dict[str, Any]) -> list[tuple[LCDocument, float | None]]:
    store = get_store(DOCS_COLLECTION)
    if cfg["search_type"] == "mmr":
        # MMR: fetch_k개를 넉넉히 가져와 질문 관련성과 서로 다른 정도를 함께 따져 k개를 고른다 (F2014)
        docs = store.max_marginal_relevance_search(
            query, k=k, fetch_k=max(cfg["mmr_fetch_k"], k), lambda_mult=cfg["mmr_lambda"], filter=where
        )
        return [(d, None) for d in docs]
    return store.similarity_search_with_score(query, k=k, filter=where)


def _shingles(text: str) -> set[tuple[str, ...]]:
    body = re.sub(r"\A\[[^\]\n]*\]\n", "", text)  # 청크 앞의 [헤더 경로] 줄은 비교에서 뺀다
    words = re.findall(r"\w+", body.lower())
    return {tuple(words[i:i + 3]) for i in range(max(len(words) - 2, 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class _Deduper:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.ids: set[str] = set()
        self.hashes: set[str] = set()
        self.shingles: list[set] = []

    def accept(self, chunk_id: str, text: str) -> bool:
        h = content_hash(text)
        if chunk_id in self.ids or h in self.hashes:
            return False
        sh = _shingles(text)
        if any(_jaccard(sh, prev) >= self.threshold for prev in self.shingles):
            return False
        self.ids.add(chunk_id)
        self.hashes.add(h)
        self.shingles.append(sh)
        return True


def _to_evidence(doc: LCDocument, score: float | None) -> Evidence:
    m = doc.metadata
    chunk_id = doc.id or f"{m.get('model_id')}:{m.get('doc_id')}:?"
    return Evidence(0, chunk_id, doc.page_content, m["source_type"], m["doc_title"], m["source_url"], m.get("page"), score)


def _take(hits: list[tuple[LCDocument, float | None]], k: int, dedup: _Deduper) -> list[Evidence]:
    taken = []
    for doc, score in hits:
        if len(taken) >= k:
            break
        e = _to_evidence(doc, score)
        if dedup.accept(e.chunk_id, e.text):
            taken.append(e)
    return taken


def retrieve(model_id: str, question: str, concept_queries: list[str] | None = None) -> list[Evidence]:
    """concept_queries: 프롬프트 1이 뽑은 개념 검색어. []면 개념 검색을 하지 않고, None(모름)이면 질문으로 검색한다."""
    cfg = get_config()["retrieval"]
    dedup = _Deduper(cfg["near_dup_threshold"])
    evidence: list[Evidence] = []

    meta = _meta_evidence(model_id)
    if meta:
        evidence.append(meta)
        dedup.accept(meta.chunk_id, meta.text)

    k = cfg["top_k"]
    evidence += _take(_search(question, {"model_id": model_id}, k * cfg["overfetch"], cfg), k, dedup)

    ck = cfg["concept_top_k"]
    queries = [question] if concept_queries is None else concept_queries
    if ck > 0 and queries:
        hits: list[tuple[LCDocument, float | None]] = []
        for q in queries:
            hits += _search(q, {"source_type": "wikipedia"}, ck * cfg["overfetch"], cfg)
        if cfg["search_type"] != "mmr":
            hits.sort(key=lambda x: x[1])  # 여러 검색어 결과를 거리순으로 합친다
        evidence += _take(hits, ck, dedup)

    for i, e in enumerate(evidence, 1):
        e.n = i
    return evidence
