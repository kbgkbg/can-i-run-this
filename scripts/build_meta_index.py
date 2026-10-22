"""모델 메타데이터 색인 (F2001~F2002).

실행: python scripts/build_meta_index.py [--fetch]
  --fetch : HF Hub에서 data/models.json을 새로 수집한 뒤 색인한다
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_config, missing_required_env  # noqa: E402
from app.data_prep import fetch_models  # noqa: E402
from app.data_prep.item_text import load_models  # noqa: E402
from app.index.indexer import index_models, reindex_cached_docs  # noqa: E402
from app.index.store import DOCS_COLLECTION, META_COLLECTION, count, embedding_mismatch, reset  # noqa: E402
from app.log import get_logger  # noqa: E402

log = get_logger("build_meta_index")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="HF Hub에서 models.json을 새로 수집")
    args = parser.parse_args()

    missing = missing_required_env()
    if missing:
        print(f".env에 다음 값이 없습니다: {', '.join(missing)} (.env.example 참고)")
        return 1

    if args.fetch or not load_models():
        print("HF Hub에서 모델 메타데이터를 수집합니다...")
        fetch_models.main()

    models = load_models()
    # 임베딩 모델이 바뀌었으면 기존 색인과 차원이 달라 섞을 수 없다: 두 컬렉션을 지우고 다시 만든다
    old_meta, old_docs = embedding_mismatch(META_COLLECTION), embedding_mismatch(DOCS_COLLECTION)
    if old_meta or old_docs:
        print(f"임베딩 모델이 바뀌었습니다 ({old_meta or old_docs} -> {get_config()['llm']['embedding_model']}). 색인을 다시 만듭니다.")
        reset(META_COLLECTION)
        if old_docs:
            reset(DOCS_COLLECTION)
            n = reindex_cached_docs()
            print(f"{DOCS_COLLECTION}: 수집해 둔 원문에서 청크 {n}개 다시 색인 (외부 재수집 없음)")
    before = count(META_COLLECTION)
    try:
        after = index_models(models)
    except Exception:
        log.exception("메타데이터 색인 실패")
        print("색인 중 오류가 발생했습니다. 자세한 내용은 logs/app.log를 확인하세요.")
        return 1

    print(f"로드한 항목: {len(models)}건")
    print(f"{META_COLLECTION} 청크 수: 색인 전 {before}건 -> 색인 후 {after}건")
    if after != len(models):
        print("경고: 항목 수와 청크 수가 다릅니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
