"""수집해 둔 문서 원문(data/cache/docs)을 현재 config.yaml의 청크 설정으로 다시 색인한다.

실행: python scripts/reindex_docs.py
외부 호출 없이(임베딩 제외) 청크 크기·중복 변경의 효과를 비교할 때 쓴다 (검증 4.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.collect import cache  # noqa: E402
from app.config import get_config  # noqa: E402
from app.index.indexer import index_document  # noqa: E402
from app.index.store import DOCS_COLLECTION, count  # noqa: E402


def main() -> int:
    cfg = get_config()["chunking"]
    before = count(DOCS_COLLECTION)
    data = cache.load()
    for model_id, entry in data.items():
        n = 0
        for doc_id in entry["docs"]:
            doc = cache.load_doc(doc_id)
            if doc:
                n += index_document(doc, model_id)
        print(f"  {model_id}: 문서 {len(entry['docs'])}건 -> 청크 {n}개")
    print(f"chunk_size={cfg['chunk_size']}, chunk_overlap={cfg['chunk_overlap']}")
    print(f"{DOCS_COLLECTION} 청크 수: {before} -> {count(DOCS_COLLECTION)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
