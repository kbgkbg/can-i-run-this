"""청킹 + 임베딩 + upsert.

청크 ID를 고정해 upsert하므로 같은 입력으로 다시 실행해도 청크 수가 늘지 않는다.
- model_meta: 항목당 1청크, ID = 모델 ID
- model_docs: ID = 모델ID:문서ID:순번 -> 여러 모델이 같은 문서를 수집해도 서로 덮어쓰지 않는다
"""
from __future__ import annotations

import hashlib
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.collect.sources.base import Document
from app.config import get_config
from app.data_prep.item_text import item_metadata, item_text
from app.index.store import DOCS_COLLECTION, META_COLLECTION, count, get_store

BATCH_SIZE = 64
MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def index_models(models: list[dict[str, Any]]) -> int:
    """모델 메타데이터를 model_meta 컬렉션에 항목당 1청크로 upsert하고, 저장된 청크 수를 반환한다."""
    store = get_store(META_COLLECTION)
    for start in range(0, len(models), BATCH_SIZE):
        batch = models[start:start + BATCH_SIZE]
        store.add_texts(
            texts=[item_text(m) for m in batch],
            metadatas=[item_metadata(m) for m in batch],
            ids=[m["id"] for m in batch],  # 청크 ID = 모델 ID
        )
    return count(META_COLLECTION)


def _size_splitter() -> RecursiveCharacterTextSplitter:
    cfg = get_config()["chunking"]
    return RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])


def split_document(doc: Document) -> list[tuple[str, int | None]]:
    """문서를 (청크 텍스트, 쪽 번호) 목록으로 나눈다. 출처 종류마다 방식을 다르게 한다.

    - hf_card: 마크다운이라 헤더(#, ##, ###) 단위로 먼저 나눈 뒤 크기 분할한다.
      모델 카드는 "Quickstart", "Evaluation", "License"처럼 절마다 주제가 분명해서,
      절 경계를 지키면 한 청크에 서로 다른 주제가 섞이지 않는다. 청크 앞에 헤더 경로를 붙여 문맥을 남긴다.
    - arxiv_pdf: 쪽 단위로 나눈 뒤 크기 분할한다. 근거에 쪽 번호를 표시해야 해서 청크가 쪽 경계를 넘지 않게 한다.
    - 그 외(wikipedia): 평문이라 크기 분할만 한다.
    """
    splitter = _size_splitter()
    if doc.pages:
        return [(chunk, page) for page, text in doc.pages for chunk in splitter.split_text(text)]

    if doc.source_type == "hf_card" and get_config()["chunking"]["markdown_header_split"]:
        sections = MarkdownHeaderTextSplitter(MD_HEADERS, strip_headers=True).split_text(doc.text)
        chunks = []
        for section in sections:
            header_path = " > ".join(section.metadata[k] for _, k in MD_HEADERS if k in section.metadata)
            for chunk in splitter.split_text(section.page_content):
                chunks.append((f"[{header_path}]\n{chunk}" if header_path else chunk, None))
        return chunks

    return [(chunk, None) for chunk in splitter.split_text(doc.text)]


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def index_document(doc: Document, model_id: str) -> int:
    """문서 1건을 모델에 연결해 model_docs에 색인하고, 만든 청크 수를 반환한다.

    같은 (모델, 문서)의 기존 청크를 먼저 지운다. 청크 크기를 바꿔 다시 색인할 때 예전 청크가 남지 않게 하기 위해서다.
    """
    store = get_store(DOCS_COLLECTION)
    store.delete(where={"$and": [{"model_id": model_id}, {"doc_id": doc.doc_id}]})

    chunks = split_document(doc)
    texts, metas, ids = [], [], []
    for i, (text, page) in enumerate(chunks):
        meta = {
            "doc_title": doc.title,
            "source_url": doc.source_url,
            "source_type": doc.source_type,
            "page": page,
            "doc_id": doc.doc_id,
            "model_id": model_id,
            "query": doc.query,
            "collected_at": doc.collected_at,
            "content_hash": content_hash(text),
        }
        texts.append(text)
        metas.append({k: v for k, v in meta.items() if v is not None})
        ids.append(f"{model_id}:{doc.doc_id}:{i}")
    for start in range(0, len(texts), BATCH_SIZE):
        store.add_texts(texts[start:start + BATCH_SIZE], metas[start:start + BATCH_SIZE], ids=ids[start:start + BATCH_SIZE])
    return len(chunks)


def count_model_docs(model_id: str) -> int:
    """선택 모델에 연결된 model_docs 청크 수."""
    try:
        return len(get_store(DOCS_COLLECTION).get(where={"model_id": model_id}, include=[])["ids"])
    except Exception:
        return 0
