"""모델 항목 1건 -> 색인 텍스트 (F2001).

모델명 · 제작사 · 파라미터 수 · 아키텍처 · 컨텍스트 길이 · 라이선스 · 지원 언어 · 태그(용도)를 한 텍스트로 묶는다.
용도·언어를 넣는 이유: 조건 기반 탐색(F2011)의 "한국어 되는 코딩용 모델" 같은 질의가 걸리게 하기 위해서다.
"""
from __future__ import annotations

import json
from typing import Any

from app.config import path

UNKNOWN = "정보 없음"

# 색인 텍스트에서 뺄 태그 (용도와 무관한 배포·출처 표시)
NOISE_TAG_PREFIXES = ("arxiv:", "base_model:", "license:", "region:", "deploy:", "dataset:", "doi:", "autotrain_compatible")
NOISE_TAGS = {"endpoints_compatible", "text-generation-inference", "safetensors", "transformers", "pytorch", "text-generation"}

# 언어 코드만 있으면 "한국어" 같은 한국어 질의가 임베딩 검색에 걸리지 않아 이름을 함께 적는다
LANG_NAMES = {
    "ko": "한국어", "en": "영어", "zh": "중국어", "ja": "일본어", "fr": "프랑스어", "de": "독일어",
    "es": "스페인어", "it": "이탈리아어", "pt": "포르투갈어", "ru": "러시아어", "ar": "아랍어",
    "hi": "힌디어", "vi": "베트남어", "th": "태국어", "id": "인도네시아어", "tr": "터키어", "nl": "네덜란드어",
}


def load_models() -> list[dict[str, Any]]:
    """data/models.json 스냅샷의 모델 목록을 읽는다. 없으면 빈 목록."""
    p = path("models_json")
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))["models"]


def format_params(n: int | None) -> str:
    if not n:
        return UNKNOWN
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if n >= size:
            return f"{n / size:.1f}{unit} ({n:,}개)"
    return f"{n:,}개"


def usage_tags(model: dict[str, Any]) -> list[str]:
    langs = set(model.get("languages") or [])
    return [
        t for t in model.get("tags", [])
        if t not in NOISE_TAGS and not t.startswith(NOISE_TAG_PREFIXES)
        and t != model.get("model_type") and t not in langs
    ]


def language_names(model: dict[str, Any]) -> list[str]:
    return [f"{LANG_NAMES[c]}({c})" if c in LANG_NAMES else c for c in model.get("languages") or []]


def item_text(model: dict[str, Any]) -> str:
    ctx = model.get("max_context")
    moe = ""
    if model.get("num_experts"):
        moe = f"MoE 전문가 {model['num_experts']}개 중 토큰당 {model.get('num_experts_per_tok') or '?'}개 활성"
    lines = [
        f"모델명: {model['id']}",
        f"제작사: {model['org']}",
        f"파라미터 수: {format_params(model.get('params'))}",
        f"아키텍처: {model.get('architecture') or UNKNOWN}" + (f" ({moe})" if moe else ""),
        f"레이어 수: {model.get('num_layers') or UNKNOWN}, KV 헤드 수: {model.get('num_kv_heads') or UNKNOWN}",
        f"컨텍스트 길이: {f'{ctx:,} 토큰' if ctx else UNKNOWN}",
        f"라이선스: {model.get('license') or UNKNOWN}",
    ]
    # 언어·태그는 값이 있을 때만 넣는다. "정보 없음" 줄이 있으면 "한국어 지원" 같은 질의에
    # 글자 모양이 비슷한 빈 항목이 오히려 상위로 걸려서다.
    optional = [
        ("지원 언어", ", ".join(language_names(model))),
        ("기반 모델", ", ".join(model.get("base_model") or [])),
        ("태그(용도)", ", ".join(usage_tags(model))),
    ]
    lines += [f"{label}: {value}" for label, value in optional if value]
    return "\n".join(lines)


def item_metadata(model: dict[str, Any]) -> dict[str, Any]:
    """Chroma 청크 메타데이터. Chroma는 None을 받지 않으므로 빈 값은 뺀다."""
    meta = {
        "model_id": model["id"],
        "doc_title": f"{model['id']} 모델 메타데이터",
        "source_url": model["hf_url"],
        "source_type": "model_meta",
        "org": model["org"],
        "params": model.get("params"),
        "license": model.get("license"),
        "languages": ",".join(model.get("languages") or []),
        "gated": bool(model.get("gated")),
    }
    return {k: v for k, v in meta.items() if v not in (None, "")}


if __name__ == "__main__":
    import sys

    models = load_models()
    target = sys.argv[1] if len(sys.argv) > 1 else models[0]["id"]
    model = next(m for m in models if m["id"] == target)
    print(item_text(model))
    print("---")
    print(item_metadata(model))
