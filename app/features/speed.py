"""토큰 생성(디코딩) 속도 추정 (F2010 특화 기능).

디코딩은 토큰 하나를 만들 때마다 가중치를 메모리에서 한 번씩 읽어야 해서 메모리 대역폭이 상한을 정한다.
  tok/s ≈ 메모리 대역폭 × 효율 계수 ÷ 토큰당 읽는 가중치 크기
- 효율 계수(speed_efficiency)는 가정값이며 실측으로 보정한다 (config.yaml).
- 일부 CPU 오프로딩: 토큰당 시간 = GPU 몫 ÷ GPU 대역폭 + RAM 몫 ÷ 시스템 RAM 대역폭
- MoE: 토큰당 활성 전문가만 읽으므로, 모델 이름의 "A3B" 같은 활성 파라미터 표기가 있으면 그 값을 쓴다.
  표기가 없으면 전체 파라미터로 계산해 "하한"으로 표시한다.
프롬프트 처리(prefill) 속도와 KV 캐시 읽기는 넣지 않았다. 결과는 추정치다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.config import get_config
from app.features.fit import CPU_ONLY, GPU_FULL, OFFLOAD, FitReport, QuantFit
from app.features.hw_detect import Hardware

GB = 1e9  # 대역폭 GB/s는 10진 기준


@dataclass
class SpeedEstimate:
    quant: str
    tok_s: float | None
    basis: str          # 어떤 값으로 계산했는지 (화면 표시용)


def active_params(model: dict[str, Any]) -> tuple[float, str]:
    """(토큰당 읽는 파라미터 수, 설명)"""
    total = model["params"]
    if not model.get("num_experts"):
        return total, "전체 파라미터"
    m = re.search(r"(?<![A-Za-z0-9])A(\d+(?:\.\d+)?)B(?![A-Za-z])", model["name"])
    if m:
        return float(m.group(1)) * 1e9, f"모델 이름의 활성 파라미터 표기(A{m.group(1)}B)"
    return total, "MoE 활성 파라미터 정보가 없어 전체 파라미터 기준(하한)"


def estimate(model: dict[str, Any], hw: Hardware, row: QuantFit) -> SpeedEstimate:
    cfg = get_config()["hardware"]
    if row.total_gb is None or row.verdict not in (GPU_FULL, OFFLOAD, CPU_ONLY):
        return SpeedEstimate(row.quant, None, "실행 불가 또는 정보 부족")
    params, basis = active_params(model)
    bytes_per_token = params * row.bpw / 8
    eff = cfg["speed_efficiency"]
    ram_bw = cfg["system_ram_bandwidth_gbps"]

    if row.verdict == CPU_ONLY:
        return SpeedEstimate(row.quant, ram_bw * GB * eff / bytes_per_token,
                             f"{basis}, 시스템 RAM 대역폭 {ram_bw:g} GB/s (가정값)")
    if hw.bandwidth_gbps is None:
        return SpeedEstimate(row.quant, None, "GPU 메모리 대역폭 정보 없음 (사양표 TODO)")
    gpu_bw = hw.bandwidth_gbps * GB * eff
    if row.verdict == GPU_FULL:
        return SpeedEstimate(row.quant, gpu_bw / bytes_per_token, f"{basis}, GPU 대역폭 {hw.bandwidth_gbps:g} GB/s")
    # 오프로딩: Apple은 같은 통합 메모리를 CPU가 읽으므로 칩 대역폭, 그 외는 시스템 RAM 대역폭
    frac = row.gpu_fraction or 0.0
    cpu_bw = gpu_bw if hw.kind == "apple" else ram_bw * GB * eff
    seconds = bytes_per_token * frac / gpu_bw + bytes_per_token * (1 - frac) / cpu_bw
    return SpeedEstimate(row.quant, 1 / seconds,
                         f"{basis}, GPU {frac:.0%} / RAM {1 - frac:.0%} 분담"
                         + ("" if hw.kind == "apple" else f" (RAM 대역폭 {ram_bw:g} GB/s 가정값)"))


def estimate_all(model: dict[str, Any], hw: Hardware, report: FitReport) -> dict[str, SpeedEstimate]:
    return {r.quant: estimate(model, hw, r) for r in report.rows}
