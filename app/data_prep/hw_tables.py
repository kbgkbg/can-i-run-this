"""GPU / Apple 칩 사양표 로드 (data/hw_specs.csv).

수치는 제조사 공식 자료에서만 채우고, 확인하지 못한 칸은 TODO로 둔다. TODO는 None으로 읽어
판정·속도 추정에서 "정보 부족"으로 처리한다.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import path


@dataclass(frozen=True)
class HWSpec:
    name: str
    type: str                     # nvidia | apple | amd | cpu
    vram_gb: float | None         # Apple은 통합 메모리라 비워 둔다
    bandwidth_gbps: float | None  # TODO면 None
    source_url: str
    note: str = ""


def _num(value: str) -> float | None:
    value = (value or "").strip()
    if not value or value.upper() == "TODO":
        return None
    return float(value)


@lru_cache(maxsize=1)
def load_specs() -> tuple[HWSpec, ...]:
    with open(path("hw_specs_csv"), encoding="utf-8", newline="") as f:
        return tuple(
            HWSpec(r["name"], r["type"], _num(r["vram_gb"]), _num(r["mem_bandwidth_gbps"]), r["source_url"], r.get("note", ""))
            for r in csv.DictReader(f)
        )


def by_name(name: str) -> HWSpec | None:
    return next((s for s in load_specs() if s.name == name), None)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# 사람들이 흔히 생략하는 제조사·브랜드 단어 ("4090", "3080 Ti", "M3 Max"처럼 말해도 맞추기 위해)
_OPTIONAL_WORDS = {"nvidia", "geforce", "rtx", "apple"}


def match_gpu(detected_name: str, vram_gb: float | None = None, gpu_cores: int | None = None) -> HWSpec | None:
    """감지된 이름(예: "NVIDIA GeForce RTX 4070", "Apple M3 Max", 대화 속 "4090")을 사양표 행에 맞춘다.

    - 사양표 이름의 단어가 모두 들어 있는 행 중 가장 긴 이름을 고른다 ("RTX 4070 Ti"가 "RTX 4070"보다 우선).
      제조사·브랜드 단어(NVIDIA, GeForce, RTX, Apple)는 없어도 된다.
    - 같은 이름에 VRAM만 다른 행(예: 4060 Ti 16GB/8GB)은 감지된 VRAM에 가까운 쪽, VRAM을 모르면 구분자 없는 기본 행.
    - Apple M3/M4 Max처럼 GPU 코어 수로 갈리는 행은 감지된 GPU 코어 수로 고른다.
    """
    if gpu_cores is None:  # "Apple M5 Max (40-core GPU)"처럼 이름에 코어 수가 들어 있으면 그 값을 쓴다
        m = re.search(r"(\d+)-core GPU", detected_name)
        gpu_cores = int(m.group(1)) if m else None
    words = set(_norm(detected_name).split())
    candidates = []
    for s in load_specs():
        base = re.sub(r"\s*\(.*?\)|\s+\d+GB$", "", s.name)  # "(40-core GPU)", " 16GB" 같은 구분자는 뺀 핵심 이름
        need = set(_norm(base).split()) - _OPTIONAL_WORDS
        if need and need <= words:
            candidates.append((len(need), s, base))
    if not candidates:
        return None
    longest = max(n for n, _, _ in candidates)
    tied = [(s, base) for n, s, base in candidates if n == longest]
    if len(tied) == 1:
        return tied[0][0]
    if gpu_cores is not None:
        for s, _ in tied:
            if f"{gpu_cores}-core GPU" in s.name:
                return s
    if vram_gb is not None:
        with_vram = [s for s, _ in tied if s.vram_gb]
        if with_vram:
            return min(with_vram, key=lambda s: abs(s.vram_gb - vram_gb))
    plain = [s for s, base in tied if s.name == base]
    return plain[0] if plain else tied[0][0]
