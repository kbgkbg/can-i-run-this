"""청킹 + 임베딩 + upsert.

청크 ID를 고정해 upsert하므로 같은 입력으로 다시 실행해도 청크 수가 늘지 않는다.
"""
from __future__ import annotations

from typing import Any

from app.data_prep.item_text import item_metadata, item_text
from app.index.store import META_COLLECTION, count, get_store

BATCH_SIZE = 64


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
