"""외부 문서 수집 단독 실행 (F2003~F2005 검증용).

실행: python scripts/collect_demo.py "Qwen/Qwen3-8B" "RTX 4070 12GB에서 Q4로 돌릴 수 있어?"
출력: 프롬프트 1 검색어, 후보, 프롬프트 2 선별/제외 문서와 이유, 색인 결과, 캐시 사용 여부
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.collect.collector import collect  # noqa: E402
from app.data_prep.item_text import load_models  # noqa: E402
from app.index.indexer import count_model_docs  # noqa: E402
from app.index.store import DOCS_COLLECTION, count  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    model_id, question = sys.argv[1], sys.argv[2]
    model = next((m for m in load_models() if m["id"] == model_id), None)
    if model is None:
        print(f"models.json에 없는 모델입니다: {model_id}")
        return 1

    t0 = time.monotonic()
    r = collect(model, question, on_step=lambda stage, msg: print(f"[{time.monotonic() - t0:5.1f}s] [{stage}] {msg}"))
    titles = {c.doc_id: f"{c.title}  <- 검색어 '{c.query}'" for c in r.candidates}

    print("\n== 프롬프트 1: 검색어 ==", "(기본값 대체)" if r.query_fallback else "")
    print(r.queries, "| 새로 수집한 검색어:", r.new_queries)
    if r.cache_hit:
        print(f"\n캐시 히트: 이미 수집된 문서 {r.docs_total}건 사용 (외부 호출 없음)")
        return 0
    print(f"\n== 후보 {len(r.candidates)}건 ==")
    print("\n== 프롬프트 2: 선별 ==")
    for i in r.selected:
        print("  [선택]", titles[i])
    for i in r.rejected:
        print("  [제외]", titles[i])
    print("  이유:", r.selection_reason)
    print("  [강제 포함]", r.forced)
    print(f"  [코드 필터로 제외: 다른 모델 저장소 {len(r.filtered)}건]", r.filtered[:6])
    print("\n== 색인 ==")
    for d in r.fetched:
        print(f"  {d.source_type:10s} {d.title} | {d.source_url} | 본문 {len(d.text):,}자")
    print(f"  이번에 만든 청크: {r.chunks_added}개 | 이 모델 청크: {count_model_docs(model_id)}개 | "
          f"{DOCS_COLLECTION} 전체: {count(DOCS_COLLECTION)}개 | 이 모델 문서: {r.docs_total}건")
    for e in r.errors:
        print("  안내:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
