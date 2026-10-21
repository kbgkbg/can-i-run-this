"""실행 가능 판정 (F2010 특화 기능).

필요 메모리 = 가중치 + KV 캐시 + 오버헤드
- 가중치 = 파라미터 수 × 양자화 bpw ÷ 8                      (파라미터 수: [근거 1] 모델 메타데이터)
- KV 캐시 = 2(K,V) × 레이어 수 × KV 헤드 수 × 헤드 차원 × 컨텍스트 × 원소 바이트 (레이어·헤드: [근거 1])
- 오버헤드 = (가중치 + KV 캐시) × overhead_ratio            (가정값, config.yaml)
판정 4단계: GPU에 전부 올라감 / 일부 CPU 오프로딩 필요 / CPU·RAM만으로 가능(느림) / 불가
모든 결과는 공식으로 낸 "계산값"이며 실제 런타임(llama.cpp, Ollama 등)에 따라 달라진다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import get_config
from app.features.hw_detect import Hardware

GIB = 1024 ** 3

GPU_FULL = "GPU에 전부 올라감"
OFFLOAD = "일부 CPU 오프로딩 필요"
CPU_ONLY = "CPU·RAM만으로 가능(느림)"
NO = "불가"
UNKNOWN = "정보 부족"


@dataclass
class QuantFit:
    quant: str
    bpw: float
    weights_gb: float
    kv_gb: float | None
    overhead_gb: float | None
    total_gb: float | None
    verdict: str
    gpu_fraction: float | None = None   # 오프로딩일 때 GPU에 올라가는 비율 (가중치 기준)


@dataclass
class FitReport:
    model_id: str
    context: int
    gpu_mem_gb: float
    ram_budget_gb: float
    rows: list[QuantFit]
    missing: list[str] = field(default_factory=list)       # 계산에 필요한데 메타데이터에 없는 항목
    alternatives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def best(self) -> QuantFit | None:
        """추천 양자화. GPU에 전부 올라가는 것이 있으면 그중 가장 정밀한 것.

        없으면(오프로딩·CPU만) 가장 낮은 bpw를 고른다. 이때는 가중치가 작을수록 GPU에 더 많이 올라가고
        RAM에서 읽는 양이 줄어 훨씬 빠르다 (예: Qwen3-8B, RTX 3080, 컨텍스트 32k에서 FP16 약 3 tok/s, Q4_K_M 수십 tok/s).
        """
        full = [r for r in self.rows if r.verdict == GPU_FULL]
        if full:
            return max(full, key=lambda r: r.bpw)
        for verdict in (OFFLOAD, CPU_ONLY):
            fits = [r for r in self.rows if r.verdict == verdict]
            if fits:
                return min(fits, key=lambda r: r.bpw)
        return None


def head_dim(model: dict[str, Any]) -> int | None:
    if model.get("head_dim"):
        return model["head_dim"]
    if model.get("hidden_size") and model.get("num_attention_heads"):
        return model["hidden_size"] // model["num_attention_heads"]
    return None


def kv_cache_bytes(model: dict[str, Any], context: int) -> float | None:
    hd = head_dim(model)
    if not (model.get("num_layers") and model.get("num_kv_heads") and hd):
        return None
    elem = get_config()["hardware"]["kv_cache_bytes"]
    return 2 * model["num_layers"] * model["num_kv_heads"] * hd * context * elem


def _verdict(total: float, hw: Hardware, ram_budget: float) -> tuple[str, float | None]:
    gpu = hw.gpu_mem_gb * GIB
    ram = ram_budget * GIB
    if hw.kind == "apple":
        # 통합 메모리: GPU 몫을 넘으면 같은 RAM에서 CPU가 나머지를 맡는다
        if total <= gpu:
            return GPU_FULL, 1.0
        if total <= ram:
            return OFFLOAD, gpu / total
        return NO, None
    if gpu > 0 and total <= gpu:
        return GPU_FULL, 1.0
    if gpu > 0 and total <= gpu + ram:
        return OFFLOAD, gpu / total
    if total <= ram:
        return CPU_ONLY, 0.0
    return NO, None


def max_context_on_gpu(model: dict[str, Any], bpw: float, hw: Hardware) -> int | None:
    """GPU 메모리에 전부 올리면서 쓸 수 있는 최대 컨텍스트 (KV 캐시를 뺀 남는 공간으로 역산)."""
    cfg = get_config()["hardware"]
    per_token = kv_cache_bytes(model, 1)
    if not per_token or not hw.gpu_mem_gb:
        return None
    weights = model["params"] * bpw / 8
    budget = hw.gpu_mem_gb * GIB / (1 + cfg["overhead_ratio"]) - weights
    return int(budget // per_token) if budget > 0 else 0


def assess(model: dict[str, Any], hw: Hardware, context: int | None = None) -> FitReport:
    cfg = get_config()["hardware"]
    context = context or cfg["default_context"]
    ram_budget = hw.ram_gb * cfg["ram_usable_ratio"]
    report = FitReport(model["id"], context, hw.gpu_mem_gb, round(ram_budget, 1), [])

    if not model.get("params"):
        report.missing.append("파라미터 수")
    kv = kv_cache_bytes(model, context)
    if kv is None:
        report.missing += [name for key, name in (("num_layers", "레이어 수"), ("num_kv_heads", "KV 헤드 수"))
                           if not model.get(key)] or ["헤드 차원"]
    if model.get("num_experts"):
        report.notes.append("MoE 모델: 메모리에는 전체 전문가 가중치가 모두 올라가야 하므로 전체 파라미터로 계산했습니다.")

    for quant, bpw in get_config()["quantization"].items():
        weights = model["params"] * bpw / 8 if model.get("params") else None
        if weights is None:
            report.rows.append(QuantFit(quant, bpw, 0, None, None, None, UNKNOWN))
            continue
        if kv is None:
            report.rows.append(QuantFit(quant, bpw, weights / GIB, None, None, None, UNKNOWN))
            continue
        overhead = (weights + kv) * cfg["overhead_ratio"]
        total = weights + kv + overhead
        verdict, frac = _verdict(total, hw, ram_budget)
        report.rows.append(QuantFit(quant, bpw, weights / GIB, kv / GIB, overhead / GIB, total / GIB, verdict, frac))

    _alternatives(model, hw, report)
    return report


def _alternatives(model: dict[str, Any], hw: Hardware, report: FitReport) -> None:
    rows = {r.quant: r for r in report.rows}
    if not rows or all(r.verdict == UNKNOWN for r in rows.values()):
        return
    precise = max(rows.values(), key=lambda r: r.bpw)
    if precise.verdict == GPU_FULL:
        return
    gpu_fits = [r for r in rows.values() if r.verdict == GPU_FULL]
    if gpu_fits:
        q = max(gpu_fits, key=lambda r: r.bpw)
        report.alternatives.append(f"낮은 양자화: {q.quant}이면 GPU에 전부 올라갑니다 (필요 {q.total_gb:.1f} GB, 계산값).")
    lowest = min(rows.values(), key=lambda r: r.bpw)
    if lowest.verdict != GPU_FULL and hw.gpu_mem_gb:
        ctx = max_context_on_gpu(model, lowest.bpw, hw)
        if ctx and ctx >= 512:
            report.alternatives.append(
                f"컨텍스트 축소: {lowest.quant}에서 컨텍스트를 약 {ctx:,} 토큰 이하로 줄이면 GPU에 올라갑니다 (계산값).")
    off = [r for r in rows.values() if r.verdict == OFFLOAD]
    if off:
        r = max(off, key=lambda r: r.bpw)
        layers = model.get("num_layers")
        on_gpu = f", 레이어 약 {int(layers * r.gpu_fraction)}/{layers}개를 GPU에" if layers and r.gpu_fraction else ""
        report.alternatives.append(
            f"CPU 오프로딩: {r.quant}은 약 {r.gpu_fraction:.0%}만 GPU에 올리고 나머지는 RAM에서 돌립니다{on_gpu} (속도 저하, 계산값).")
    if all(r.verdict in (NO, UNKNOWN) for r in rows.values()):
        report.alternatives.append("이 모델은 현재 메모리로는 어떤 양자화로도 올라가지 않습니다. 더 작은 모델을 고려하세요.")


def calc_text(model: dict[str, Any], hw: Hardware, report: FitReport, speeds: dict[str, float | None] | None = None) -> str:
    """프롬프트 3에 넣을 [계산값] 블록. 답변이 판정을 인용할 때 "(계산값)"으로 표시하게 한다."""
    cfg = get_config()["hardware"]
    lines = [
        f"하드웨어: {hw.gpu_name or 'GPU 없음'} (GPU 사용 가능 {hw.gpu_mem_gb:.1f} GB), RAM {hw.ram_gb:.1f} GB "
        f"(모델용 {report.ram_budget_gb:.1f} GB로 계산), 메모리 대역폭 "
        + (f"{hw.bandwidth_gbps:g} GB/s (출처 {hw.bandwidth_source})" if hw.bandwidth_gbps else "정보 없음"),
        f"계산식: 필요 메모리 = 파라미터 수 × bpw ÷ 8 + KV 캐시(2 × 레이어 × KV 헤드 × 헤드 차원 × 컨텍스트 {report.context:,} × "
        f"{cfg['kv_cache_bytes']}바이트) + 오버헤드 {cfg['overhead_ratio']:.0%}. 파라미터 수·레이어 수·KV 헤드 수는 [근거 1] 값.",
    ]
    if report.missing:
        lines.append(f"정보 부족: {', '.join(report.missing)}이(가) 메타데이터에 없어 판정할 수 없습니다.")
    for r in report.rows:
        if r.total_gb is None:
            continue
        speed = (speeds or {}).get(r.quant)
        sp = f", 추정 속도 약 {speed:.0f} tok/s" if speed else ""
        lines.append(f"- {r.quant} (bpw {r.bpw}): 가중치 {r.weights_gb:.1f} GB + KV {r.kv_gb:.2f} GB + 오버헤드 "
                     f"{r.overhead_gb:.1f} GB = {r.total_gb:.1f} GB -> {r.verdict}{sp}")
    lines += [f"대안: {a}" for a in report.alternatives]
    lines += report.notes
    return "\n".join(lines)
