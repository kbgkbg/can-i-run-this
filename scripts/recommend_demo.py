"""조건 기반 모델 탐색 단독 실행 (F2011 검증용).

실행: python scripts/recommend_demo.py "한국어 되는 코딩용 모델" [--gpu "RTX 3080" --ram 32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features.hw_detect import manual  # noqa: E402
from app.features.recommend import search  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("condition")
    p.add_argument("--gpu")
    p.add_argument("--ram", type=float, default=32)
    a = p.parse_args()
    hw = manual(a.gpu, a.ram) if a.gpu else None
    r = search(a.condition, hw)
    print("해석한 조건:", r.condition)
    print(f"후보 {r.candidates}건 | 코드 필터 제외 {r.excluded} | 근거 불일치로 버림 {r.rejected_evidence}건")
    for rec in r.results:
        fit = f" | {rec.quant}: {rec.verdict} ({rec.total_gb:.1f} GB" + (f", 약 {rec.tok_s:.0f} tok/s" if rec.tok_s else "") + ")" if rec.quant else (f" | {rec.verdict}" if rec.verdict else "")
        print(f"  - {rec.model_id}{fit}\n      이유: {rec.reason}\n      근거: {rec.evidence or '(코드 필터 조건)'}")
    if not r.results:
        print("  조건에 맞는 모델이 없습니다.")
    for n in r.notes:
        print("  참고:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
