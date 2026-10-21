"""유사도 검색 vs MMR 검색 근거 비교 (F2014 검증용).

실행: python scripts/compare_mmr.py [--top-k 8]
같은 질문마다 세 방식으로 근거를 뽑아 비교하고 docs/eval/mmr_comparison.md로 저장한다.
  A. similarity, 근사 중복 제거 끔   (가장 단순한 유사도 검색)
  B. similarity, 근사 중복 제거 켬
  C. mmr (fetch_k, lambda는 config.yaml), 근사 중복 제거 켬   (현재 기본 설정)
지표 (메타데이터 근거 1건 제외한 외부 문서 근거 기준)
  - 서로 다른 문서 수: 클수록 여러 문서에서 고르게 가져왔다
  - 근거 간 평균 코사인 유사도: 낮을수록 서로 겹치는 내용이 적다 (임베딩은 Chroma에 저장된 값)
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_config  # noqa: E402
from app.index.store import DOCS_COLLECTION, get_store  # noqa: E402
from app.rag.retrieve import Evidence, retrieve  # noqa: E402

QUESTIONS = [
    ("Qwen/Qwen3-8B", "Qwen3-8B의 구조(레이어, 어텐션 헤드)와 컨텍스트 길이는?"),
    ("Qwen/Qwen3-8B", "Qwen3는 사전 학습에 어떤 데이터를 얼마나 썼어?"),
    ("Qwen/Qwen3-8B", "thinking 모드와 non-thinking 모드의 권장 샘플링 설정은?"),
    ("Qwen/Qwen3-4B", "긴 텍스트를 처리하려면 어떻게 설정해야 해?"),
]

SETTINGS = [
    ("A. similarity (중복 제거 끔)", {"search_type": "similarity", "near_dup_threshold": 1.01}),
    ("B. similarity + 근사 중복 제거", {"search_type": "similarity"}),
    ("C. MMR + 근사 중복 제거 (기본)", {"search_type": "mmr"}),
]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def metrics(evidence: list[Evidence]) -> tuple[int, float | None]:
    docs = [e for e in evidence if e.source_type != "model_meta"]
    distinct = len({(e.title, e.page) if e.source_type == "arxiv_pdf" else e.title for e in docs})
    distinct_docs = len({e.title for e in docs})
    if len(docs) < 2:
        return distinct_docs, None
    got = get_store(DOCS_COLLECTION).get(ids=[e.chunk_id for e in docs], include=["embeddings"])
    vecs = dict(zip(got["ids"], got["embeddings"]))
    pairs = [(_cos(vecs[a.chunk_id], vecs[b.chunk_id])) for a, b in itertools.combinations(docs, 2)
             if a.chunk_id in vecs and b.chunk_id in vecs]
    _ = distinct  # 쪽 단위 구분은 표의 근거 목록에서 확인한다
    return distinct_docs, (sum(pairs) / len(pairs) if pairs else None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=8)
    a = p.parse_args()
    cfg = get_config()["retrieval"]
    base = dict(cfg)
    cfg["top_k"] = a.top_k
    cfg["concept_top_k"] = 0  # 개념 문서는 빼고 model_docs 검색만 비교한다

    lines = [
        "# 유사도 검색 vs MMR 검색 비교 (F2014)", "",
        f"- top_k={a.top_k}, MMR fetch_k={cfg['mmr_fetch_k']}, lambda={cfg['mmr_lambda']}, "
        f"근사 중복 기준 자카드 {base['near_dup_threshold']}, 청크 {get_config()['chunking']['chunk_size']}/"
        f"{get_config()['chunking']['chunk_overlap']}자",
        "- 지표는 메타데이터 근거([근거 1])를 뺀 외부 문서 근거 기준. 평균 코사인 유사도는 낮을수록 근거끼리 덜 겹친다.",
        "- 생성: `python scripts/compare_mmr.py`", "",
    ]
    summary = ["| 질문 | " + " | ".join(name for name, _ in SETTINGS) + " |", "|---|" + "---|" * len(SETTINGS)]

    for model_id, q in QUESTIONS:
        lines += [f"## {model_id} — {q}", ""]
        row = [q]
        for name, override in SETTINGS:
            cfg.update(base, top_k=a.top_k, concept_top_k=0, **override)
            ev = retrieve(model_id, q, concept_queries=[])
            n_docs, sim = metrics(ev)
            row.append(f"문서 {n_docs}종 · 유사도 {sim:.3f}" if sim is not None else f"문서 {n_docs}종")
            lines += [f"**{name}** — 서로 다른 문서 {n_docs}종, 근거 간 평균 코사인 유사도 "
                      + (f"{sim:.3f}" if sim is not None else "-"), ""]
            for e in ev[1:]:
                head = e.text.split("\n", 1)[0][:70].replace("|", "/")
                lines.append(f"- {e.tag} {e.title} · `{e.chunk_id.rsplit(':', 1)[-1]}번 청크` · {head}")
            lines.append("")
        summary.append("| " + " | ".join(row) + " |")
        print(f"[완료] {q}")

    out = PROJECT_ROOT / "docs" / "eval" / "mmr_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines[:6] + ["## 요약", ""] + summary + [""] + lines[6:]), encoding="utf-8")
    print("\n".join(summary))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
