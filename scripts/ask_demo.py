"""질문 -> (수집) -> retrieve -> generate 전체 실행 (F2006~F2008 검증용).

실행: python scripts/ask_demo.py "Qwen/Qwen3-8B" "질문" [--top-k 3] [--search-type mmr] [--no-collect]
  --top-k, --search-type : config.yaml retrieval 값을 이번 실행에서만 바꿔 근거 변화를 비교한다
  --no-collect           : 외부 문서 수집을 건너뛰고 이미 색인된 문서만으로 답한다
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.collect.collector import collect, extract_queries, route  # noqa: E402
from app.config import get_config  # noqa: E402
from app.data_prep.item_text import load_models  # noqa: E402
from app.rag.citations import invalid_citations  # noqa: E402
from app.rag.graph import run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("question")
    p.add_argument("--top-k", type=int)
    p.add_argument("--search-type", choices=["similarity", "mmr"])
    p.add_argument("--no-collect", action="store_true")
    a = p.parse_args()

    cfg = get_config()["retrieval"]
    if a.top_k:
        cfg["top_k"] = a.top_k
    if a.search_type:
        cfg["search_type"] = a.search_type
    model = next((m for m in load_models() if m["id"] == a.model_id), None)
    if model is None:
        print(f"models.json에 없는 모델입니다: {a.model_id}")
        return 1

    t0 = time.monotonic()
    if a.no_collect:
        queries, _ = extract_queries(model, a.question)
        concept = [q for q in queries if route(q, model) == "wiki"]
        print(f"[수집 생략] 검색어 {queries} | 개념 검색어 {concept}")
    else:
        r = collect(model, a.question)
        concept = r.concept_queries
        print(f"[수집] 검색어 {r.queries} | 개념 검색어 {concept} | "
              + (f"이미 수집된 문서 {r.docs_total}건 사용" if r.cache_hit else f"새 문서 {len(r.fetched)}건"))
    t1 = time.monotonic()
    state = run(a.model_id, a.question, concept_queries=concept)
    t2 = time.monotonic()

    print(f"\n== 근거 목록 (top_k={cfg['top_k']}, concept_top_k={cfg['concept_top_k']}, {cfg['search_type']}) ==")
    for e in state["evidence"]:
        score = "" if e.score is None else f" dist={e.score:.3f}"
        print(f"  {e.tag} {e.title} | {e.chunk_id}{score}")
    print("\n== 답변 ==\n" + state["answer"])
    print("\n== 화면에 표시할 근거 (실제 인용된 것만) ==")
    for e in state["cited"]:
        print(f"  {e.tag} {e.title} — {e.url}")
    if state["notice"]:
        print("  " + state["notice"])
    for u in state.get("unverified", []):
        print(f"  ⚠ 근거에서 확인되지 않은 수치: {u}")
    bad = invalid_citations(state["answer"], state["evidence"])
    if bad:
        print(f"  경고: 근거 목록에 없는 번호 인용 {bad}")
    print(f"\n소요: 수집 {t1 - t0:.1f}s, 검색+생성 {t2 - t1:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
