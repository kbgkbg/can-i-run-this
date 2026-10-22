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
    # 코사인 거리: OpenAI 임베딩은 정규화돼 있어 코사인이 일반적인 선택이다.
    # 어떤 임베딩 모델로 만든 컬렉션인지 기록해 둔다 (모델을 바꾸면 차원이 달라 섞을 수 없다)
    return Chroma(
        collection_name=name,
        embedding_function=embeddings(),
        client=client(),
        collection_metadata={"hnsw:space": "cosine", "embedding_model": get_config()["llm"]["embedding_model"]},
    )


def embedding_mismatch(name: str) -> str | None:
    """컬렉션이 지금 설정과 다른 임베딩 모델로 만들어졌으면 그 모델 이름(기록이 없으면 "이전 버전"), 아니면 None."""
    try:
        meta = client().get_collection(name).metadata or {}
    except Exception:
        return None
    built = meta.get("embedding_model")
    return None if built == get_config()["llm"]["embedding_model"] else (built or "이전 버전(text-embedding-3-small)")


def reset(name: str) -> None:
    """컬렉션을 지운다 (색인은 원문 캐시·스냅샷에서 다시 만들 수 있는 파생 데이터다)."""
    try:
        client().delete_collection(name)
    except Exception:
        pass


def count(name: str) -> int:
    """컬렉션 청크 수. 컬렉션이 없으면 0 (임베딩 호출 없이 조회)."""
    try:
        return client().get_collection(name).count()
    except Exception:
        return 0
