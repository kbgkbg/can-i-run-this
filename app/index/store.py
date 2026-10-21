"""Chroma 컬렉션 생성/조회.

- model_meta: 모델 메타데이터, 항목당 1청크, 청크 ID = 모델 ID
- model_docs: 외부 문서 청크, 청크 ID = 모델ID:문서ID:순번
"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import env, get_config, path

META_COLLECTION = "model_meta"
DOCS_COLLECTION = "model_docs"


@lru_cache(maxsize=1)
def embeddings() -> OpenAIEmbeddings:
    cfg = get_config()["llm"]
    return OpenAIEmbeddings(
        model=cfg["embedding_model"],
        openai_api_key=env("OPENAI_API_KEY"),
        openai_api_base=env("OPENAI_BASE_URL"),
        # 토큰 ID 배열 대신 원문 문자열을 보낸다 (tiktoken 인코딩 파일 다운로드 불필요, GMS 호환)
        check_embedding_ctx_length=False,
        max_retries=cfg["max_retries"],
        request_timeout=cfg["request_timeout_s"],
    )


@lru_cache(maxsize=1)
def client() -> chromadb.ClientAPI:
    chroma_dir = path("chroma_dir")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_store(name: str) -> Chroma:
    # 코사인 거리: OpenAI 임베딩은 정규화돼 있어 코사인이 일반적인 선택이다
    return Chroma(
        collection_name=name,
        embedding_function=embeddings(),
        client=client(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def count(name: str) -> int:
    """컬렉션 청크 수. 컬렉션이 없으면 0 (임베딩 호출 없이 조회)."""
    try:
        return client().get_collection(name).count()
    except Exception:
        return 0
