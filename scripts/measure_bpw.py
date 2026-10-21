"""공식 GGUF 저장소의 실제 파일 크기로 양자화별 실효 bpw(가중치당 비트 수)를 측정한다.

실행: python scripts/measure_bpw.py
실효 bpw = GGUF 파일 크기(바이트) × 8 ÷ 원본 모델 파라미터 수 (models.json의 safetensors 기준)
GGUF 파일에는 토크나이저·메타데이터도 들어 있어 아주 약간 크게 나온다. 결과는 config.yaml quantization에 반영한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.collect import http  # noqa: E402
from app.config import path  # noqa: E402

BASES = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


def main() -> int:
    models = {m["id"]: m for m in json.loads(path("models_json").read_text(encoding="utf-8"))["models"]}
    worst: dict[str, float] = {}
    for base in BASES:
        params = models[base]["params"]
        files = http.get_json(f"https://huggingface.co/api/models/{base}-GGUF/tree/main")
        row = {}
        for f in files:
            for q in QUANTS:
                if f["path"].endswith(f"-{q}.gguf"):
                    row[q] = f["size"] * 8 / params
                    worst[q] = max(worst.get(q, 0.0), row[q])
        print(f"{base}-GGUF  " + "  ".join(f"{q}={row[q]:.3f}" for q in QUANTS if q in row))
    print("최댓값(보수적 판정용): " + "  ".join(f"{q}={v:.3f}" for q, v in worst.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
