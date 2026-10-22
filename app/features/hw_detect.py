"""로컬 하드웨어 감지 (F2010 특화 기능의 입력).

Streamlit 서버가 사용자 PC에서 돌기 때문에 서버가 직접 사양을 읽는다.
- 공통: psutil로 전체 RAM, CPU 이름·코어 수
- NVIDIA: pynvml(nvidia-ml-py)로 GPU 이름·VRAM, 실패하면 nvidia-smi 출력 파싱
- Apple Silicon: sysctl로 칩 이름·메모리, system_profiler로 GPU 코어 수.
  통합 메모리라 GPU 사용 가능 메모리 = RAM × apple_gpu_mem_ratio
- 그 외/실패: kind="cpu" (화면에서 수동 입력으로 보완)

단독 실행: python -m app.features.hw_detect  -> JSON 출력
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field

import psutil

from app.config import get_config
from app.data_prep.hw_tables import HWSpec, by_name, match_gpu
from app.log import get_logger

log = get_logger("hw_detect")
GIB = 1024 ** 3


@dataclass
class Hardware:
    kind: str                         # nvidia | apple | cpu
    cpu_name: str
    cpu_cores: int | None
    ram_gb: float
    gpu_name: str | None = None
    vram_gb: float | None = None      # 전용 VRAM (Apple은 None)
    gpu_mem_gb: float = 0.0           # 모델이 GPU에서 쓸 수 있는 메모리 (Apple: RAM × 비율)
    gpu_cores: int | None = None      # Apple GPU 코어 수 (M3/M4 Max 구분용)
    spec_name: str | None = None      # 사양표에서 맞춘 행
    bandwidth_gbps: float | None = None
    bandwidth_source: str | None = None
    method: str = ""                  # 감지 방법 (pynvml | nvidia-smi | sysctl | manual | none)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Hardware":
        return cls(**d)


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "알 수 없음"


def _nvidia() -> tuple[str, float, str] | None:
    """(GPU 이름, VRAM GiB, 방법). 여러 장이면 첫 번째 GPU."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                return name, pynvml.nvmlDeviceGetMemoryInfo(h).total / GIB, "pynvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001 - 드라이버 없음 등. nvidia-smi로 다시 시도
        log.info("pynvml 감지 실패: %r", e)
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if out and out.strip():
        name, mib = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return name, float(mib) / 1024, "nvidia-smi"
    return None


def _apple_gpu_cores() -> int | None:
    out = _run(["system_profiler", "SPDisplaysDataType"])
    m = re.search(r"Total Number of Cores:\s*(\d+)", out or "")
    return int(m.group(1)) if m else None


def _apply_spec(hw: Hardware, spec: HWSpec | None) -> Hardware:
    if spec is None:
        hw.notes.append("사양표에 없는 장치라 메모리 대역폭을 알 수 없습니다 (속도 추정 불가).")
        return hw
    hw.spec_name = spec.name
    hw.bandwidth_gbps = spec.bandwidth_gbps
    hw.bandwidth_source = spec.source_url
    if spec.bandwidth_gbps is None:
        hw.notes.append(f"사양표의 {spec.name} 대역폭이 TODO(공식 출처 미확인)라 속도를 추정하지 않습니다.")
    return hw


def detect() -> Hardware:
    cfg = get_config()["hardware"]
    ram_gb = psutil.virtual_memory().total / GIB
    base = dict(cpu_name=_cpu_name(), cpu_cores=psutil.cpu_count(logical=False), ram_gb=round(ram_gb, 1))

    nv = _nvidia()
    if nv:
        name, vram, method = nv
        hw = Hardware(kind="nvidia", gpu_name=name, vram_gb=round(vram, 1), gpu_mem_gb=round(vram, 1), method=method, **base)
        return _apply_spec(hw, match_gpu(name, vram_gb=vram))

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            ram_gb = int(out.strip()) / GIB
            base["ram_gb"] = round(ram_gb, 1)
        cores = _apple_gpu_cores()
        hw = Hardware(kind="apple", gpu_name=base["cpu_name"], gpu_cores=cores,
                      gpu_mem_gb=round(ram_gb * cfg["apple_gpu_mem_ratio"], 1), method="sysctl", **base)
        hw.notes.append(f"통합 메모리: GPU 사용 가능량을 RAM × {cfg['apple_gpu_mem_ratio']}로 계산했습니다.")
        return _apply_spec(hw, match_gpu(base["cpu_name"], gpu_cores=cores))

    hw = Hardware(kind="cpu", method="none", **base)
    hw.notes.append("지원하는 GPU를 찾지 못했습니다. GPU가 있다면 직접 입력하세요.")
    return hw


def manual(spec_name: str | None, ram_gb: float, apple_gpu_ratio: float | None = None,
           bandwidth_gbps: float | None = None) -> Hardware:
    """수동 입력: 사양표에서 고른 GPU·칩(없으면 CPU만) + RAM. Mac은 GPU에 쓸 통합 메모리 비율을 바꿀 수 있다.

    bandwidth_gbps: 사양표 대역폭이 TODO인 장치에 사용자가 직접 넣은 값. 사양표 값이 있으면 무시한다.
    """
    cfg = get_config()["hardware"]
    ratio = apple_gpu_ratio or cfg["apple_gpu_mem_ratio"]
    base = dict(cpu_name=_cpu_name(), cpu_cores=psutil.cpu_count(logical=False), ram_gb=float(ram_gb), method="manual")
    spec = by_name(spec_name) if spec_name else None
    if spec is None:
        return Hardware(kind="cpu", **base)
    if spec.type == "apple":
        hw = Hardware(kind="apple", gpu_name=spec.name, gpu_mem_gb=round(ram_gb * ratio, 1), **base)
        hw.notes.append(f"통합 메모리: GPU 사용 가능량을 RAM × {ratio:g}로 계산했습니다.")
    else:
        hw = Hardware(kind=spec.type, gpu_name=spec.name, vram_gb=spec.vram_gb, gpu_mem_gb=spec.vram_gb or 0.0, **base)
    hw = _apply_spec(hw, spec)
    if hw.bandwidth_gbps is None and bandwidth_gbps:
        hw.bandwidth_gbps = float(bandwidth_gbps)
        hw.bandwidth_source = "사용자 입력"
        hw.notes = [n for n in hw.notes if "TODO" not in n]
        hw.notes.append(f"메모리 대역폭 {bandwidth_gbps:g} GB/s는 사용자가 입력한 값입니다 (공식 출처 미확인).")
    return hw


if __name__ == "__main__":
    print(json.dumps(detect().to_dict(), ensure_ascii=False, indent=2))
