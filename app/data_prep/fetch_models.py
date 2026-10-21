"""HF Hub에서 text-generation 모델 메타데이터를 수집해 data/models.json에 스냅샷으로 저장한다 (F2001).

실행: python -m app.data_prep.fetch_models

수집 규칙
- 목록: /api/models?pipeline_tag=text-generation&sort=downloads (다운로드순)
- 파라미터 수: 목록 응답의 safetensors.total (safetensors 메타데이터 기준)
- 레이어 수·KV 헤드 등: 각 모델의 config.json. gated이거나 없으면 null ("정보 부족")
- 제외: 파라미터 수를 알 수 없는 저장소(GGUF 전용 등), 양자화 파생 저장소, 테스트용 초소형 모델
  -> 양자화 저장소는 safetensors.total이 압축된 텐서 수로 잡혀 원본 파라미터 수와 다르다.
     이 서비스는 원본 파라미터 수에 양자화 비트를 곱해 직접 계산하므로 원본 모델만 항목으로 둔다.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from app.collect import http
from app.config import env, get_config, path
from app.log import get_logger

log = get_logger("fetch_models")

HF_API = "https://huggingface.co/api/models"
LIST_PAGE_SIZE = 200
MAX_LIST_PAGES = 10

# 양자화 파생 저장소 판별: 저장소 이름으로만 본다.
# 태그(fp8, mxfp4, onnx 등)는 gpt-oss, DeepSeek-V3처럼 원본이 그 형식으로 배포된 모델에도 붙어 있어 쓰지 않는다.
QUANT_PATTERN = re.compile(
    r"(gguf|awq|gptq|exl2|nvfp4|mxfp4|fp8|int4|int8|bnb|4bit|8bit|mlx|onnx|quant)", re.IGNORECASE
)
TEST_ORG_PATTERN = re.compile(r"(internal-testing|tiny-random|^hf-internal)", re.IGNORECASE)
MIN_PARAMS = 50_000_000  # 이보다 작으면 테스트·실험용 모델로 보고 제외


def _hf_headers() -> dict[str, str]:
    token = env("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _list_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """다운로드순 목록을 cursor 페이지네이션으로 가져온다."""
    params = [
        ("pipeline_tag", cfg["pipeline_tag"]),
        ("sort", cfg["sort"]),
        ("direction", "-1"),
        ("limit", str(LIST_PAGE_SIZE)),
    ] + [("expand[]", f) for f in ("safetensors", "cardData", "gated", "downloads", "lastModified", "tags", "config")]

    url: str | None = HF_API
    items: list[dict[str, Any]] = []
    for _ in range(MAX_LIST_PAGES):
        resp = http.get(url, params=params if url == HF_API else None, headers=_hf_headers())
        items.extend(resp.json())
        url = resp.links.get("next", {}).get("url")
        if not url or len(_filter(items)) >= cfg["target_count"]:
            break
    return items


def _exclude_reason(m: dict[str, Any]) -> str | None:
    model_id = m["id"]
    params = (m.get("safetensors") or {}).get("total")
    if TEST_ORG_PATTERN.search(model_id):
        return "테스트용 저장소"
    if QUANT_PATTERN.search(model_id.split("/", 1)[-1]):
        return "양자화 파생 저장소"
    if not params:
        return "파라미터 수 없음(safetensors 메타데이터 없음)"
    if params < MIN_PARAMS:
        return "초소형 모델"
    return None


def _filter(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in items if _exclude_reason(m) is None]


def _fetch_config(model_id: str) -> tuple[dict[str, Any] | None, str]:
    """config.json을 받아 (config, 상태)를 반환한다. 상태: ok | gated_or_missing"""
    try:
        resp = http.get(f"https://huggingface.co/{model_id}/resolve/main/config.json", headers=_hf_headers())
        return resp.json(), "ok"
    except (http.ExternalAPIError, ValueError) as e:
        log.info("config.json 없음/접근 불가: %s (%s)", model_id, e)
        return None, "gated_or_missing"


def _pick(cfg: dict[str, Any] | None, *keys: str) -> Any:
    """config에서 첫 번째로 존재하는 키 값을 반환. 멀티모달 모델은 text_config 아래도 본다."""
    if not cfg:
        return None
    for scope in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {}):
        for k in keys:
            if scope.get(k) is not None:
                return scope[k]
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _build_item(m: dict[str, Any], config: dict[str, Any] | None, config_status: str) -> dict[str, Any]:
    model_id = m["id"]
    card = m.get("cardData") or {}
    listing_cfg = m.get("config") or {}
    num_heads = _pick(config, "num_attention_heads", "n_head", "num_heads")
    hidden = _pick(config, "hidden_size", "n_embd", "d_model")
    head_dim = _pick(config, "head_dim")
    if head_dim is None and hidden and num_heads:
        head_dim = hidden // num_heads
    org, _, name = model_id.partition("/")
    return {
        "id": model_id,
        "org": org,
        "name": name or model_id,
        "params": m["safetensors"]["total"],
        "params_by_dtype": m["safetensors"].get("parameters"),
        "architecture": (_as_list(_pick(config, "architectures")) or _as_list(listing_cfg.get("architectures")) or [None])[0],
        "model_type": _pick(config, "model_type") or listing_cfg.get("model_type"),
        "num_layers": _pick(config, "num_hidden_layers", "n_layer", "num_layers"),
        "hidden_size": hidden,
        "num_attention_heads": num_heads,
        # num_key_value_heads가 없으면 MHA로 보고 attention head 수와 같다고 둔다 (transformers 기본 동작)
        "num_kv_heads": _pick(config, "num_key_value_heads", "num_kv_heads") or num_heads,
        "head_dim": head_dim,
        "max_context": _pick(config, "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"),
        "num_experts": _pick(config, "num_local_experts", "num_experts", "n_routed_experts"),
        "num_experts_per_tok": _pick(config, "num_experts_per_tok", "experts_per_token"),
        "license": card.get("license"),
        "license_link": card.get("license_link"),
        "languages": _as_list(card.get("language")),
        "base_model": _as_list(card.get("base_model")),
        "tags": m.get("tags", []),
        "gated": m.get("gated") or False,
        "config_status": config_status,
        "downloads": m.get("downloads"),
        "last_modified": m.get("lastModified"),
        "hf_url": f"https://huggingface.co/{model_id}",
    }


def fetch(target_count: int | None = None) -> dict[str, Any]:
    cfg = dict(get_config()["fetch_models"])
    if target_count:
        cfg["target_count"] = target_count

    listed = _list_models(cfg)
    excluded: dict[str, int] = {}
    kept = []
    for m in listed:
        reason = _exclude_reason(m)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        elif len(kept) < cfg["target_count"]:
            kept.append(m)

    models = []
    for i, m in enumerate(kept, 1):
        config, status = _fetch_config(m["id"])
        models.append(_build_item(m, config, status))
        if i % 25 == 0:
            print(f"  config.json {i}/{len(kept)}")

    return {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{HF_API}?pipeline_tag={cfg['pipeline_tag']}&sort={cfg['sort']}",
        "listed_count": len(listed),
        "excluded": excluded,
        "count": len(models),
        "models": models,
    }


def main() -> None:
    snapshot = fetch()
    out = path("models_json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")

    models = snapshot["models"]
    missing = sum(1 for m in models if m["num_layers"] is None)
    print(f"목록 {snapshot['listed_count']}건 중 {snapshot['count']}건 저장 -> {out}")
    print(f"제외: {snapshot['excluded']}")
    print(f"config.json 정보 부족(gated/없음): {missing}건")


if __name__ == "__main__":
    main()
