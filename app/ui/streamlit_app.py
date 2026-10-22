"""Streamlit UI (F2009, F2010, F2011, F2012).

실행: streamlit run app/ui/streamlit_app.py
- 사이드바: 모델 선택, 조건으로 모델 찾기, 하드웨어 감지/직접 입력, 색인 현황(메타데이터 건수, 선택 모델의 수집 문서)
- 메인: 조건 탐색 결과 카드 + 선택 모델 헤더 + 실행 가능 판정 카드 + 모델별로 분리된 대화 + 진행 상태 + 인용된 근거
화면에는 안내 문구만 띄우고 프롬프트 전문·스택 트레이스는 보이지 않는다 (NF2004).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # streamlit run은 스크립트 폴더만 경로에 넣는다

import psutil  # noqa: E402
import streamlit as st  # noqa: E402

from app.collect import cache  # noqa: E402
from app.config import get_config, missing_required_env, path  # noqa: E402
from app.data_prep.hw_tables import load_specs  # noqa: E402
from app.data_prep.item_text import format_params, load_models  # noqa: E402
from app.features import hw_detect  # noqa: E402
from app.features.fit import FitReport, assess, calc_text, context_sweep, max_context_on_gpu, memory_text  # noqa: E402
from app.features.hw_detect import Hardware  # noqa: E402
from app.features import consult, persona  # noqa: E402
from app.features.recommend import Recommendation, search as recommend_search  # noqa: E402
from app.features.speed import SpeedEstimate, estimate, estimate_all  # noqa: E402
from app.index.indexer import count_model_docs  # noqa: E402
from app.index.store import META_COLLECTION, DOCS_COLLECTION, count, embedding_mismatch  # noqa: E402
from app.log import get_logger  # noqa: E402
from app.rag.pipeline import ask, user_message  # noqa: E402

log = get_logger("ui")
st.set_page_config(page_title="Can I run this model?", page_icon="🖥️", layout="wide")

MODE_ASK, MODE_CONSULT = "모델에게 묻기", "추천 상담"
FIRST_ASK_NOTICE = ("⏳ 이 모델에 처음 질문하면 Hugging Face·위키백과·arXiv에서 근거 문서를 찾아 모으느라 "
                    "수십 초 걸릴 수 있습니다. 같은 모델의 다음 질문부터는 모아 둔 문서를 다시 써서 빨라집니다.")
EXAMPLE_QUESTIONS = ["Q4로 돌리면 메모리 얼마나 필요해?", "권장 샘플링 설정은?", "상업적으로 써도 돼?"]
PERSONA_EXAMPLES = ["능금이 뭐야?", "사과 5개 중 2개 먹고 3개 더 사서 반을 친구 주면 몇 개 남아?",
                    "파이썬으로 리스트에서 두 번째로 큰 수 구하는 함수 짜줘"]
# 질문 입력창이 화면 아래에서 잘 안 보여 말풍선처럼 테두리·그림자로 구분한다 (모드 공통)
CHAT_INPUT_CSS = """<style>
/* 테두리는 회색 안쪽 상자에 건다. 바깥 상자에 걸면 안쪽 상자(모서리 8px)가 둥근 테두리 밖으로 삐져나와 모서리가 잘려 보인다 */
[data-testid="stChatInput"] > div:first-child {
  border: 2px solid var(--primary-color, #ff4b4b) !important;
  border-radius: 1.4rem !important;
  box-shadow: 0 4px 16px rgba(255, 75, 75, 0.22);
}
[data-testid="stChatInput"] > div:first-child:focus-within { box-shadow: 0 4px 22px rgba(255, 75, 75, 0.38); }
</style>"""
VERDICT_ICON = {"GPU에 전부 올라감": "🟢", "일부 CPU 오프로딩 필요": "🟡", "CPU·RAM만으로 가능(느림)": "🟠", "불가": "🔴", "정보 부족": "⚪"}


@st.cache_data(show_spinner=False)
def models_by_id() -> dict[str, dict]:
    return {m["id"]: m for m in load_models()}


SORT_POPULAR, SORT_TRENDING, SORT_FIT, SORT_RECENT, SORT_SMALL = (
    "🔥 인기순 (다운로드)", "📈 요즘 뜨는 순", "⭐ 내 PC 추천순", "🆕 최근 업데이트순", "📏 작은 모델부터")


@st.cache_data(show_spinner=False)
def trending_rank() -> dict[str, int]:
    """스냅샷의 trending 목록(HF trendingScore 순) -> 순위 (1부터)."""
    snap = json.loads(path("models_json").read_text(encoding="utf-8"))
    return {mid: i for i, mid in enumerate(snap.get("trending", []), 1)}


@st.cache_data(show_spinner=False)
def fit_ranking(hw_json: str) -> dict[str, tuple[int, float, str]]:
    """내 PC 추천순: 모델 ID -> (구간, -파라미터, 표시 문구). 구간이 작을수록 위.

    0: GPU에 다 올라가고 쓸 만한 속도(usable_tok_s 이상) / 1: GPU에 다 올라가지만 느림 / 2: 오프로딩 / 3: CPU만 / 4: 불가·정보 부족.
    같은 구간에서는 큰 모델(더 똑똑할 가능성이 큰)부터 보인다.
    """
    hw = Hardware.from_dict(json.loads(hw_json))
    floor = get_config()["hardware"]["usable_tok_s"]
    out = {}
    for mid, m in models_by_id().items():
        report = assess(m, hw)
        speeds = {q: sp.tok_s for q, sp in estimate_all(m, hw, report).items()}
        best = report.best(speeds)
        if best is None or report.missing:
            out[mid] = (4, 0.0, "불가" if best is None and not report.missing else "정보 부족")
            continue
        tok = speeds.get(best.quant)
        tier = {"GPU에 전부 올라감": 0 if (tok or 0) >= floor else 1, "일부 CPU 오프로딩 필요": 2}.get(best.verdict, 3)
        out[mid] = (tier, -(m.get("params") or 0), f"{VERDICT_ICON[best.verdict]} {best.quant}" + (f" {tok:.0f} tok/s" if tok else ""))
    return out


def sorted_model_ids(models: dict[str, dict], order: str, hw: Hardware | None) -> tuple[list[str], dict[str, str]]:
    """(정렬된 모델 ID, 목록에 붙일 표시 문구)"""
    ids = list(models)  # models.json 순서 = 다운로드순
    if order == SORT_TRENDING:
        rank = trending_rank()
        ids.sort(key=lambda i: (rank.get(i, 10 ** 6), -(models[i].get("downloads") or 0)))
        return ids, {i: f"📈 {rank[i]}위" for i in ids if i in rank}
    if order == SORT_FIT and hw is not None:
        fit = fit_ranking(json.dumps(hw.to_dict(), sort_keys=True))
        ids.sort(key=lambda i: fit[i][:2])
        return ids, {i: fit[i][2] for i in ids}
    if order == SORT_RECENT:
        ids.sort(key=lambda i: models[i].get("last_modified") or "", reverse=True)
        return ids, {i: f"🆕 {(models[i].get('last_modified') or '')[:10]}" for i in ids}
    if order == SORT_SMALL:
        ids.sort(key=lambda i: models[i].get("params") or 0)
        return ids, {}
    return ids, {i: f"↓{_compact(models[i].get('downloads'))}" for i in ids if models[i].get("downloads")}


def _compact(n: int | None) -> str:
    if not n:
        return "?"
    for unit, size in (("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def md_escape(text: str) -> str:
    """모델 이름의 _ * 등이 마크다운 문법으로 해석되지 않게 한다 (예: Q4_K_M)."""
    return "".join("\\" + ch if ch in "\\`*_[]<>#|" else ch for ch in text)


def short_params(n: int | None) -> str:
    return format_params(n).split(" ")[0] if n else "?"


def chats() -> dict[str, list[dict]]:
    return st.session_state.setdefault("chats", {})


def current_hw() -> Hardware | None:
    d = st.session_state.get("hw")
    return Hardware.from_dict(d) if d else None


# ------------------------------------------------------------------ 비정상 상황 안내 (F2012)
def preflight() -> dict[str, dict] | None:
    missing = missing_required_env()
    if missing:
        st.error(f"실행에 필요한 설정이 없습니다: {', '.join(missing)}\n\n"
                 "`.env.example`을 `.env`로 복사한 뒤 값을 채우고 앱을 다시 실행하세요.")
        return None
    models = models_by_id()
    if not models or count(META_COLLECTION) == 0:
        st.warning("모델 메타데이터 색인이 비어 있습니다 (0건).\n\n"
                   "터미널에서 `python scripts/build_meta_index.py`를 먼저 실행하세요.")
        return None
    old = embedding_mismatch(META_COLLECTION) or embedding_mismatch(DOCS_COLLECTION)
    if old:
        st.warning(f"임베딩 모델이 바뀌어({old} → {get_config()['llm']['embedding_model']}) 기존 색인을 쓸 수 없습니다.\n\n"
                   "터미널에서 `python scripts/build_meta_index.py`를 실행하면 수집해 둔 문서까지 다시 색인합니다 (재수집 없음).")
        return None
    return models


# ------------------------------------------------------------------ 사이드바
MAC_MEMORY_GB = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]  # M3 Ultra는 512GB까지  # 선택지 (Mac 구성마다 고를 수 있는 값이 다르다)


def _detect_hw() -> None:
    try:
        st.session_state["hw"] = hw_detect.detect().to_dict()
    except Exception:  # noqa: BLE001
        log.exception("하드웨어 감지 실패")
        st.warning("하드웨어를 자동으로 감지하지 못했습니다. 아래에서 직접 고르세요.")


def _bw_label(spec) -> str:
    return f"{spec.bandwidth_gbps:g} GB/s" if spec.bandwidth_gbps else "대역폭 미확인"


def _bandwidth_input(spec: Any, key: str) -> float | None:
    """사양표에 대역폭이 없는(TODO) 장치면 사용자가 확인한 값을 직접 넣게 한다. 서비스가 값을 지어내지는 않는다."""
    if spec.bandwidth_gbps is not None:
        return None
    bw = st.number_input("메모리 대역폭 (GB/s, 선택)", 0.0, 10000.0, 0.0, 1.0, key=key,
                         help="공식 출처를 확인하지 못해 사양표에 비어 있는 값입니다. 제조사 사양 페이지나 GPU-Z 등에서 "
                              "확인한 값을 넣으면 속도 추정·체험이 켜집니다. 넣은 값은 '사용자 입력값'으로 표시됩니다.")
    st.caption("ℹ 이 장치는 메모리 대역폭이 사양표에 없어(공식 출처 미확인) 속도를 추정하지 못합니다. 알고 있다면 위에 입력하세요.")
    return bw or None


def spec_picker(prefix: str) -> None:
    """내 PC 사양 선택. Mac 사용자가 많아 Apple Silicon을 별도 탭으로 둔다 (통합 메모리라 입력 방식이 다르다)."""
    specs = load_specs()
    default_ram = int(round(psutil.virtual_memory().total / 1024 ** 3))
    mac, nv, cpu = st.tabs(["🍎 Mac (Apple Silicon)", "🟩 NVIDIA GPU", "💻 GPU 없음"])
    with mac:
        chips = [x for x in specs if x.type == "apple"]
        chip = st.selectbox("칩", chips, key=f"{prefix}_mac_chip", format_func=lambda x: f"{x.name} · {_bw_label(x)}")
        mem = st.selectbox("통합 메모리", MAC_MEMORY_GB, index=MAC_MEMORY_GB.index(16), key=f"{prefix}_mac_mem",
                           format_func=lambda g: f"{g} GB")
        ratio = st.slider("GPU가 쓸 수 있는 메모리 비율", 0.5, 0.9, float(get_config()["hardware"]["apple_gpu_mem_ratio"]),
                          0.05, key=f"{prefix}_mac_ratio",
                          help="통합 메모리 중 모델(GPU)에 쓸 수 있다고 보는 비율. 기본값은 가정값이며 나머지는 OS·앱 몫으로 둔다.")
        st.caption(f"GPU 사용 가능 약 {mem * ratio:.1f} GB로 판정합니다.")
        mac_bw = _bandwidth_input(chip, f"{prefix}_mac_bw")
        if st.button("Mac 사양 적용", key=f"{prefix}_mac_apply", width="stretch"):
            st.session_state["hw"] = hw_detect.manual(chip.name, mem, ratio, bandwidth_gbps=mac_bw).to_dict()
            st.rerun()
    with nv:
        gpus = [x for x in specs if x.type == "nvidia"]
        gpu = st.selectbox("GPU", gpus, key=f"{prefix}_nv_gpu",
                           format_func=lambda x: f"{x.name} · VRAM {x.vram_gb:g} GB · {_bw_label(x)}")
        ram = st.number_input("시스템 RAM (GB)", 1, 2048, default_ram, key=f"{prefix}_nv_ram")
        nv_bw = _bandwidth_input(gpu, f"{prefix}_nv_bw")
        if st.button("NVIDIA 사양 적용", key=f"{prefix}_nv_apply", width="stretch"):
            st.session_state["hw"] = hw_detect.manual(gpu.name, ram, bandwidth_gbps=nv_bw).to_dict()
            st.rerun()
    with cpu:
        ram_c = st.number_input("시스템 RAM (GB)", 1, 2048, default_ram, key=f"{prefix}_cpu_ram")
        st.caption("GPU 없이 CPU와 RAM만으로 돌리는 경우입니다. 속도는 시스템 RAM 대역폭(가정값)으로 추정합니다.")
        if st.button("CPU만 사용으로 적용", key=f"{prefix}_cpu_apply", width="stretch"):
            st.session_state["hw"] = hw_detect.manual(None, ram_c).to_dict()
            st.rerun()


def onboarding() -> None:
    """첫 화면: 판정·추천이 모두 하드웨어에 달려 있어 먼저 내 PC 사양을 고르게 한다."""
    with st.container(border=True):
        st.subheader("🖥️ 먼저 내 PC 사양을 알려 주세요")
        st.caption("실행 가능 판정, 속도 추정, 조건 탐색의 'GPU에서 돌아가는지'가 모두 이 사양으로 계산됩니다. 사이드바에서 언제든 바꿀 수 있습니다.")
        c1, c2 = st.columns(2)
        if c1.button("🔍 자동 감지 (NVIDIA·Apple Silicon)", key="onb_detect", width="stretch"):
            _detect_hw()
            st.rerun()
        if c2.button("나중에 할게요", key="onb_skip", width="stretch"):
            st.session_state["hw_skip"] = True
            st.rerun()
        spec_picker("onb")


def hardware_panel() -> None:
    st.header("내 하드웨어")
    if st.button("하드웨어 감지", width="stretch"):
        _detect_hw()
    with st.expander("사양 선택 / 변경"):
        spec_picker("side")

    hw = current_hw()
    if hw is None:
        st.caption("하드웨어를 감지하거나 고르면 실행 가능 판정이 표시됩니다.")
        return
    if hw.kind == "apple":
        st.caption(f"**{hw.gpu_name}** · 통합 메모리 {hw.ram_gb:.0f} GB (GPU 사용 {hw.gpu_mem_gb:.1f} GB)")
    elif hw.gpu_name:
        st.caption(f"**{hw.gpu_name}** · VRAM {hw.vram_gb:.1f} GB · RAM {hw.ram_gb:.0f} GB")
    else:
        st.caption(f"GPU 없음 · RAM {hw.ram_gb:.0f} GB · {hw.cpu_name}")
    if hw.bandwidth_gbps:
        src = hw.bandwidth_source or ""
        st.caption(f"메모리 대역폭 {hw.bandwidth_gbps:g} GB/s · "
                   + (f"[사양 출처]({src})" if src.startswith("http") else "사용자 입력값 (공식 출처 미확인)"))
    st.caption(f"감지 방법: {hw.method}")
    for n in hw.notes:
        st.caption(f"ℹ {n}")


def sidebar(models: dict[str, dict]) -> tuple[dict | None, Any]:
    """(선택 모델, 수집 문서 목록을 나중에 채울 사이드바 자리)"""
    with st.sidebar:
        st.radio("모드", [MODE_ASK, MODE_CONSULT], key="mode", horizontal=True,
                 help="추천 상담: 챗봇과 대화하며 조건을 말하면 맞는 모델을 찾아 줍니다.")
        st.header("모델 선택")
        order = st.selectbox("정렬", [SORT_POPULAR, SORT_TRENDING, SORT_FIT, SORT_RECENT, SORT_SMALL], key="model_sort",
                             help="⭐ 내 PC 추천순: 고른 사양에서 GPU에 다 올라가고 쓸 만한 속도가 나는 모델을 큰 것부터 보여 줍니다.")
        hw = current_hw()
        if order == SORT_FIT and hw is None:
            st.caption("내 PC 사양을 먼저 고르면 추천순으로 정렬됩니다. 지금은 인기순입니다.")
        ids, badge = sorted_model_ids(models, order, hw)
        model_id = st.selectbox(
            "질문할 모델", ids, index=None, key="model_id",
            format_func=lambda i: f"{i}  ·  {short_params(models[i].get('params'))}" + (f"  ·  {badge[i]}" if i in badge else ""),
            placeholder="모델 이름을 입력해 검색하세요",
        )
        with st.expander("🔎 조건으로 모델 찾기"):
            st.text_area("원하는 조건", key="rec_query", height=80,
                         placeholder="예: 한국어 되는 코딩용 모델, 내 그래픽카드에서 돌아가는 거")
            if st.button("찾기", width="stretch") and st.session_state.get("rec_query", "").strip():
                run_recommend(st.session_state["rec_query"].strip())
        st.divider()
        hardware_panel()
        st.divider()
        st.caption(f"모델 메타데이터 색인: {count(META_COLLECTION)}건")
        # 수집 문서 목록은 자리만 잡고 main() 끝에서 채운다. 사이드바가 질문 처리보다 먼저 그려져
        # 방금 수집한 문서가 다음 조작 전까지 "0건"으로 보이던 문제를 막는다.
        docs_slot = st.empty()
        return (models.get(model_id) if model_id else None), docs_slot


def render_doc_status(m: dict, slot: Any) -> None:
    with slot.container():
        docs = cache.indexed_docs(m["id"])
        st.caption(f"수집된 문서 {len(docs)}건 · 청크 {count_model_docs(m['id'])}개")
        for d in docs.values():
            st.caption(f"· [{md_escape(d['title'])}]({d['source_url']})")
        if m.get("config_status") != "ok":
            st.caption("⚠ 이 모델은 gated 저장소라 레이어 수 등 일부 메타데이터가 없습니다.")


# ------------------------------------------------------------------ 조건 기반 탐색 (F2011)
def run_recommend(text: str) -> None:
    with st.spinner("조건에 맞는 모델을 찾는 중..."):
        try:
            st.session_state["rec"] = {"query": text, "result": recommend_search(text, current_hw())}
        except Exception:  # noqa: BLE001
            log.exception("조건 탐색 실패")
            st.session_state["rec"] = {"query": text, "error": "조건 탐색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}


def _choose_model(model_id: str) -> None:
    st.session_state["model_id"] = model_id  # 카드를 누르면 그 모델을 선택해 대화로 이어간다
    st.session_state["mode"] = MODE_ASK


def render_rec_card(rec_item: Recommendation, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{md_escape(rec_item.model_id)}**")
        if rec_item.quant:
            sp = f", 약 {rec_item.tok_s:.0f} tok/s" if rec_item.tok_s else ""
            st.caption(f"{VERDICT_ICON.get(rec_item.verdict, '')} {rec_item.quant}: {rec_item.verdict} "
                       f"({rec_item.total_gb:.1f} GB{sp}, 계산값)")
        elif rec_item.verdict:
            st.caption(f"{VERDICT_ICON.get(rec_item.verdict, '')} {rec_item.verdict}")
        st.write(rec_item.reason)
        if rec_item.evidence:
            st.caption(f"근거 (모델 메타데이터): “{md_escape(rec_item.evidence)}” · [원문]({rec_item.url})")
        else:
            st.caption(f"근거 (모델 메타데이터): 언어·크기 표기 · [원문]({rec_item.url})")
        st.button("이 모델로 대화", key=key, on_click=_choose_model, args=(rec_item.model_id,), width="stretch")


def recommend_cards() -> None:
    rec = st.session_state.get("rec")
    if not rec:
        return
    with st.expander(f"🔎 조건 탐색 결과: {rec['query']}", expanded=True):
        if rec.get("error"):
            st.warning(rec["error"])
            return
        r = rec["result"]
        c = r.condition
        st.caption(f"해석한 조건 — 언어: {', '.join(c.languages) or '무관'} · 용도: {', '.join(c.uses) or '무관'} · "
                   f"파라미터: {c.min_params_b or '-'}~{c.max_params_b or '-'}B · GPU 적재 필수: {'예' if c.must_fit_gpu else '아니오'}"
                   f" | 후보 {r.candidates}건 중 조건 불일치 {sum(r.excluded.values())}건 제외")
        if not r.results:
            st.info("조건을 모두 만족하는 모델을 찾지 못했습니다. 조건을 줄여 다시 찾아보세요.")
        cols = st.columns(2)
        for i, rec_item in enumerate(r.results):
            with cols[i % 2]:
                render_rec_card(rec_item, key=f"pick_{rec_item.model_id}")
        for n in r.notes:
            st.caption(f"ℹ {n}")


# ------------------------------------------------------------------ 메인
def model_header(m: dict) -> None:
    st.subheader(m["id"])
    cols = st.columns(4)
    cols[0].metric("파라미터", short_params(m.get("params")))
    cols[1].metric("컨텍스트", f"{m['max_context']:,}" if m.get("max_context") else "정보 없음")
    cols[2].metric("레이어", m.get("num_layers") or "정보 없음")
    cols[3].metric("라이선스", m.get("license") or "정보 없음")
    st.caption(f"[Hugging Face 모델 페이지]({m['hf_url']})")


def fit_card(m: dict, hw: Hardware) -> tuple[FitReport, dict[str, SpeedEstimate]]:
    """실행 가능 판정 카드. 판정에 쓴 값의 근거(메타데이터·사양표)와 계산값을 구분해 보여 준다."""
    cfg = get_config()["hardware"]
    max_ctx = int(m.get("max_context") or 131072)
    min_ctx = min(512, max_ctx)
    # 모델마다 최대 컨텍스트가 달라 값을 모델별 키로 둔다. 한 키를 공유하면 32k로 올린 뒤
    # 4k 모델로 바꿨을 때 위젯 값이 최댓값을 넘어 예외가 났다.
    # 화면에 없는 위젯의 상태는 Streamlit이 지우므로, 모델별 값은 일반 딕셔너리에 따로 보관해 돌아왔을 때 복원한다.
    key = f"context::{m['id']}"
    saved = st.session_state.setdefault("contexts", {})
    if key not in st.session_state:
        st.session_state[key] = saved.get(m["id"], min(cfg["default_context"], max_ctx))
    context = max(min_ctx, min(int(st.session_state[key]), max_ctx))
    st.session_state[key] = saved[m["id"]] = context
    report = assess(m, hw, context)
    speeds = estimate_all(m, hw, report)

    with st.container(border=True):
        best = report.best({q: sp.tok_s for q, sp in speeds.items()})
        if report.missing:
            hint = " (gated 모델은 .env에 HF_TOKEN을 넣고 메타데이터를 다시 수집하면 채워집니다)" if m.get("gated") else ""
            st.markdown(f"⚪ **정보 부족** — {', '.join(report.missing)}이(가) 메타데이터에 없어 판정할 수 없습니다{hint}.")
        elif best is None:
            st.markdown("🔴 **불가** — 현재 메모리로는 어떤 양자화로도 올라가지 않습니다 (계산값).")
        else:
            sp = speeds[best.quant].tok_s
            st.markdown(f"{VERDICT_ICON[best.verdict]} **{best.quant}: {best.verdict}** — 필요 {best.total_gb:.1f} GB"
                        + (f", 추정 약 {sp:.0f} tok/s" if sp else "") + " (계산값)")
            precise = max((r for r in report.rows if r.verdict == best.verdict == "GPU에 전부 올라감"), key=lambda r: r.bpw, default=None)
            if precise and precise.quant != best.quant and speeds[precise.quant].tok_s:
                st.caption(f"{precise.quant}도 GPU에 올라가지만 약 {speeds[precise.quant].tok_s:.0f} tok/s로 느려, "
                           f"쓸 만한 속도({get_config()['hardware']['usable_tok_s']} tok/s 이상, 가정값)가 나오는 {best.quant}을 추천합니다.")

        c1, c2 = st.columns([1, 3])
        c1.number_input("컨텍스트 길이", min_value=min_ctx, max_value=max_ctx, step=1024, key=key)
        rows = [{
            "양자화": r.quant,
            "필요 메모리(GB)": None if r.total_gb is None else round(r.total_gb, 1),
            "가중치(GB)": round(r.weights_gb, 1) if r.weights_gb else None,
            "KV 캐시(GB)": None if r.kv_gb is None else round(r.kv_gb, 2),
            "판정": f"{VERDICT_ICON[r.verdict]} {r.verdict}",
            "추정 속도(tok/s)": None if speeds[r.quant].tok_s is None else round(speeds[r.quant].tok_s, 1),
        } for r in report.rows]
        c2.dataframe(rows, hide_index=True, width="stretch")

        for a in report.alternatives:
            st.caption(f"💡 {a}")
        context_table(m, hw, report, speeds)
        with st.expander("판정 근거와 계산식"):
            st.markdown(
                f"- 파라미터 수 {format_params(m.get('params'))}, 레이어 {m.get('num_layers') or '정보 없음'}, "
                f"KV 헤드 {m.get('num_kv_heads') or '정보 없음'} — **[근거 1] (모델 메타데이터)** · [원문]({m['hf_url']})\n"
                + ((f"- 메모리 대역폭 {hw.bandwidth_gbps:g} GB/s — **하드웨어 사양표** · [출처]({hw.bandwidth_source})\n"
                    if (hw.bandwidth_source or "").startswith("http") else
                    f"- 메모리 대역폭 {hw.bandwidth_gbps:g} GB/s — **사용자 입력값** (공식 출처 미확인)\n")
                   if hw.bandwidth_gbps else "- 메모리 대역폭: 정보 없음 (속도 추정 생략)\n")
                + f"- **계산값**: 필요 메모리 = 파라미터 × bpw ÷ 8 + KV 캐시(2 × 레이어 × KV 헤드 × 헤드 차원 × 컨텍스트 × "
                  f"{cfg['kv_cache_bytes']}바이트) + 오버헤드 {cfg['overhead_ratio']:.0%}\n"
                + f"- **계산값**: 속도 ≈ 메모리 대역폭 × 효율 {cfg['speed_efficiency']} ÷ 토큰당 읽는 가중치 크기 "
                  f"({speeds[report.rows[0].quant].basis})\n"
                + "- 양자화 bpw와 효율·오버헤드 등 가정값은 `app/config.yaml`에 출처와 함께 있습니다. "
                  "실제 메모리 사용량과 속도는 런타임(llama.cpp, Ollama 등)과 설정에 따라 다릅니다."
            )
            for n in report.notes:
                st.caption(f"ℹ {n}")
    return report, speeds


def context_table(m: dict, hw: Hardware, report: FitReport, speeds: dict[str, SpeedEstimate]) -> None:
    """컨텍스트 길이를 늘리면 KV 캐시·필요 메모리·판정·속도가 얼마나 달라지는지 (모델 최대 컨텍스트까지)."""
    if report.missing:
        return
    with st.expander("📏 컨텍스트 길이별 차이"):
        quants = [r.quant for r in report.rows]
        best = report.best({q: sp.tok_s for q, sp in speeds.items()})
        q = st.selectbox("양자화", quants, index=quants.index(best.quant) if best else 0, key=f"ctx_quant::{m['id']}")
        bpw = get_config()["quantization"][q]
        rows = []
        for ctx, row in context_sweep(m, hw, q):
            sp = estimate(m, hw, row).tok_s
            rows.append({"컨텍스트": f"{ctx:,}", "KV 캐시(GB)": round(row.kv_gb, 2), "필요 메모리(GB)": round(row.total_gb, 1),
                         "판정": f"{VERDICT_ICON[row.verdict]} {row.verdict}",
                         "추정 속도(tok/s)": None if sp is None else round(sp, 1)})
        st.dataframe(rows, hide_index=True, width="stretch")
        fit_max = max_context_on_gpu(m, bpw, hw)
        if fit_max:
            st.caption(f"{q}로 GPU에 전부 올리면서 쓸 수 있는 최대 컨텍스트: 약 {min(fit_max, m.get('max_context') or fit_max):,} 토큰 (계산값)")
        st.caption("KV 캐시는 컨텍스트에 비례해 커지고, 디코딩 때 매 토큰 KV 캐시도 읽어서 길수록 느려진다. "
                   "속도는 컨텍스트를 가득 채운 상태 기준(보수적)이며 추정치다.")


def render_sources(msg: dict) -> None:
    cited = msg.get("cited", [])
    if not cited:
        st.caption(msg.get("notice") or "이 답변에는 인용된 근거가 없습니다.")
        return
    with st.expander(f"답변에 사용한 근거 {len(cited)}건"):
        for e in cited:
            st.markdown(f"**[근거 {e['n']}]** {md_escape(e['title'])} · {e['label']} · [원문]({e['url']})")


def _paced(text: str, tok_s: float):
    """답변을 추정 속도(tok/s)로 흘려 보낸다. 글자 수 ÷ stream_chars_per_token을 토큰 수로 근사한다."""
    step = max(1, int(get_config()["hardware"]["stream_chars_per_token"]))
    delay = 1.0 / tok_s
    for i in range(0, len(text), step):
        yield text[i:i + step]
        time.sleep(delay)


def _speed_caption(msg: dict, stream_tok_s: float | None) -> None:
    """추정 속도 안내. 재생할 때와 지난 대화로 다시 그릴 때 요소 구성을 똑같이 맞춘다.

    재생 때만 캡션을 넣었더니 다음 질문을 처리하는 동안 요소 위치가 한 칸씩 밀려, 이전 실행의 답변이
    흐린 잔상으로 한 번 더 보였다 (Streamlit은 실행이 끝날 때까지 이전 요소를 남겨 둔다).
    """
    if stream_tok_s:
        msg["played_tok_s"] = stream_tok_s
        st.caption(f"⏱ 내 PC 추정 속도 {stream_tok_s:.1f} tok/s로 재생 중 (추정치, 실제와 다를 수 있음)")
    elif msg.get("played_tok_s"):
        st.caption(f"⏱ 내 PC 추정 속도 {msg['played_tok_s']:.1f} tok/s로 재생했습니다 (추정치)")


def render_message(msg: dict, stream_tok_s: float | None = None) -> None:
    if msg.get("persona"):
        # 🎭 대화 체험의 답은 연기 답변 말풍선으로만 보인다
        render_persona(msg, stream_tok_s)
        return
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and not msg.get("error"):
            _speed_caption(msg, stream_tok_s)
        if stream_tok_s and msg["role"] == "assistant" and not msg.get("error"):
            st.write_stream(_paced(msg["content"], stream_tok_s))
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and not msg.get("error"):
            render_sources(msg)
            if msg.get("unverified"):
                st.warning("인용한 근거에서 확인되지 않은 수치가 있습니다. 원문을 직접 확인하세요.\n\n"
                           + "\n".join(f"- {md_escape(u)}" for u in msg["unverified"]))
            for w in msg.get("warnings", []):
                st.caption(f"ℹ {w}")
            if msg.get("elapsed"):
                st.caption(f"처리 시간 {msg['elapsed']:.1f}초")


def render_persona(msg: dict, stream_tok_s: float | None = None) -> None:
    """🎭 연기 답변 말풍선."""
    pa = msg["persona"]
    with st.chat_message("🎭", avatar="🎭"):
        st.markdown(f"**{md_escape(pa['model_id'])} · {pa['quant']} · {pa['tier']} 체급(유효 약 {pa['params_b']}B)** (연기)")
        _speed_caption(msg, stream_tok_s)
        if stream_tok_s:
            st.write_stream(_paced(pa["answer"], stream_tok_s))
        else:
            st.markdown(pa["answer"])


def first_question(m: dict) -> bool:
    """이 모델의 문서를 아직 하나도 수집하지 않았는지 (첫 질문은 수집 때문에 오래 걸린다)."""
    return not cache.indexed_docs(m["id"])


def welcome(m: dict, persona_mode: bool = False) -> None:
    """대화가 비어 있을 때: 어디에 무엇을 물으면 되는지 말풍선으로 안내하고 예시 질문을 버튼으로 준다."""
    if persona_mode:
        with st.chat_message("🎭", avatar="🎭"):
            st.markdown(f"**{md_escape(m['name'])}**(연기)와 대화해 보세요. 문서 검색 없이 아무 주제나 됩니다. "
                        "상식·계산·코딩을 물어보면 이 크기 모델의 수준이 잘 드러납니다.")
            cols = st.columns(len(PERSONA_EXAMPLES))
            for col, q in zip(cols, PERSONA_EXAMPLES):
                col.button(q, key=f"pexample::{q}", width="stretch",
                           on_click=lambda q=q: st.session_state.__setitem__("pending_question", q))
            st.caption(persona.PERSONA_NOTICE)
        return
    with st.chat_message("assistant"):
        st.markdown(f"**{md_escape(m['name'])}**에 대해 물어보세요. 화면 아래 💬 **질문 입력창**에 쓰거나 예시를 눌러도 됩니다.")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            col.button(q, key=f"example::{q}", width="stretch",
                       on_click=lambda q=q: st.session_state.__setitem__("pending_question", q))
        if first_question(m):
            st.caption(FIRST_ASK_NOTICE)


def history_for_prompt(history: list[dict], turns: int = 3) -> list[tuple[str, str]]:
    """같은 모델과의 최근 대화 (오류 답변 제외). 이어지는 질문("그럼 Q5는?")을 이해하는 데 쓴다."""
    msgs = [(x["role"], x["content"]) for x in history if x["role"] in ("user", "assistant") and not x.get("error")]
    return msgs[-turns * 2:]


def handle_question(m: dict, question: str, calc: str, history: list[tuple[str, str]] | None = None) -> dict:
    first = first_question(m)
    label = "질문을 처리하는 중... (이 모델의 첫 질문이라 근거 문서를 모으고 있어 시간이 더 걸립니다)" \
        if first else "질문을 처리하는 중..."
    with st.status(label, expanded=True) as status:
        result = ask(m, question, calc=calc, history=history,
                     on_step=lambda stage, text: st.write(f"**{stage}** · {text}"))
        c = result.collect
        if c is not None:
            if c.cache_hit:
                st.write(f"이미 수집된 문서 {c.docs_total}건 사용")
            elif c.fetched:
                st.write("새로 수집한 문서: " + ", ".join(d.title for d in c.fetched))
        if result.ok:
            done = "답변 완료" + (f" · 첫 질문이라 근거 문서 수집 포함 {sum(result.timings.values()):.0f}초" if first else "")
            status.update(label=done, state="complete", expanded=False)
        else:
            status.update(label="답변을 만들지 못했습니다", state="error", expanded=False)

    if not result.ok:
        return {"role": "assistant", "content": f"⚠ {result.error}", "error": True}
    return {
        "role": "assistant",
        "content": result.answer,
        "cited": [{"n": e.n, "title": e.title, "label": e.label, "url": e.url} for e in result.cited],
        "notice": result.notice,
        "unverified": result.unverified,
        "warnings": result.warnings,
        "elapsed": sum(result.timings.values()),
    }


def experience_controls(m: dict, report: FitReport | None, speeds: dict[str, SpeedEstimate]) -> tuple[float | None, str | None]:
    """체험 토글: '내 PC 속도로 체험'(추정 tok/s로 재생)과 '🎭 흉내 답변'(이 모델 체급의 답 연기).

    반환: (재생 속도 또는 None, 흉내 답변에 쓸 양자화 또는 None)
    """
    usable = [r.quant for r in (report.rows if report else []) if speeds.get(r.quant) and speeds[r.quant].tok_s]
    quants = usable or list(get_config()["quantization"])
    best = report.best({q: sp.tok_s for q, sp in speeds.items()}) if report else None
    default = quants.index(best.quant) if best and best.quant in quants else quants.index("Q4_K_M") if "Q4_K_M" in quants else 0
    c1, c2, c3 = st.columns([1, 1, 1])
    speed_on = c1.toggle("내 PC 속도로 체험", key="speed_exp",
                         help="답변을 이 PC에서 모델을 돌렸을 때의 추정 생성 속도로 흘려 보여 줍니다. "
                              "사양을 아직 안 골랐으면 켤 때 이 PC 하드웨어를 자동으로 감지합니다.")
    if speed_on and current_hw() is None:
        # 모델만 직접 고르고 사양은 건너뛴 경우: 켜는 순간 이 PC를 감지해 바로 쓸 수 있게 한다
        _detect_hw()
        st.rerun()
    persona_on = c2.toggle("🎭 이 모델과 대화해 보기", key="persona_exp",
                           help="문서 검색 없이, 이 모델 체급(파라미터 크기)의 로컬 챗봇이라면 어떻게 답할지 연기합니다. "
                                "아무 주제나 말을 걸어 '이 정도 지능이면 쓸 만한지' 체감해 보세요. 실제 모델 출력이 아닙니다.")
    # 키에 추천 양자화를 넣어, 컨텍스트·모델 변경으로 추천이 바뀌면 기본값도 따라가게 한다
    # (한 키를 쓰면 위젯이 처음 값을 기억해 4k에서의 추천 Q8_0이 32k에서도 남았다)
    quant = c3.selectbox("체험 양자화", quants, index=default, key=f"speed_quant::{m['id']}::{quants[default]}",
                         label_visibility="collapsed", disabled=not (speed_on or persona_on))
    if speed_on and not usable:
        hw = current_hw()
        if hw.kind != "cpu" and hw.bandwidth_gbps is None:
            reason = (f"⏱ {hw.spec_name or hw.gpu_name or '이 장치'}의 메모리 대역폭이 사양표에 없어(공식 출처 미확인) 속도를 재생할 수 없습니다. "
                      "사이드바 '사양 선택 / 변경'에서 대역폭을 직접 입력하면 켜집니다.")
        else:
            reason = "⏱ 이 모델은 현재 사양에서 실행할 수 있는 양자화가 없어 속도를 추정하지 않습니다."
        c1.caption(reason)
    tok_s = speeds[quant].tok_s if speed_on and quant in usable else None
    return tok_s, (quant if persona_on else None)


def persona_reply(m: dict, quant: str, question: str, history: list[tuple[str, str]]) -> dict:
    """🎭 대화 체험: 문서 검색 없이 이 모델 체급의 대화 상대를 연기한 답."""
    with st.spinner(f"{m['name']}처럼 생각하는 중... (연기)"):
        try:
            pa = persona.answer(m, quant, question, history=history)
        except Exception as e:  # noqa: BLE001
            log.exception("흉내 답변 실패")
            return {"role": "assistant", "content": f"⚠ {user_message(e)}", "error": True}
    return {"role": "assistant", "content": pa.answer,
            "persona": {"model_id": m["id"], "quant": quant, "tier": pa.persona.tier, "params_b": pa.persona.params_b,
                        "answer": pa.answer, "mistakes": pa.mistakes, "removed": pa.removed_numbers, "added": pa.added_numbers}}


# ------------------------------------------------------------------ 추천 상담 (F2011 대화형)
def consult_page() -> None:
    st.subheader("💬 모델 추천 상담")
    st.caption("용도·언어·크기·하드웨어를 말하면 조건에 맞는 모델을 메타데이터 근거와 함께 찾아 드립니다. "
               "예: \"코딩용인데 한국어도 되면 좋겠어. 그래픽카드는 RTX 3080, 램 32기가\"")
    history = st.session_state.setdefault("consult", [])
    for i, msg in enumerate(history):
        render_consult_message(msg, i)
    text = st.chat_input("원하는 모델 조건을 말해 주세요")
    if not text:
        return
    history.append({"role": "user", "content": text})
    render_consult_message(history[-1], len(history) - 1)
    with st.spinner("조건을 정리하는 중..."):
        try:
            asked = 0  # 마지막 검색 이후 되물은 횟수
            for h in reversed(history[:-1]):
                if h["role"] == "assistant":
                    if "cards" in h:
                        break
                    asked += 1
            turn = consult.step([{"role": h["role"], "content": h["content"]} for h in history], current_hw(), asked)
        except Exception as e:  # noqa: BLE001
            log.exception("추천 상담 실패")
            history.append({"role": "assistant", "content": f"⚠ {user_message(e)}"})
            render_consult_message(history[-1], len(history) - 1)
            return
    reply: dict = {"role": "assistant", "content": turn.message, "notes": [turn.hw_note] if turn.hw_note else []}
    if turn.hardware is not None:
        st.session_state["hw"] = turn.hardware.to_dict()
    if turn.action == "search":
        with st.spinner("조건에 맞는 모델을 찾는 중..."):
            try:
                r = recommend_search(turn.condition, current_hw())
            except Exception as e:  # noqa: BLE001
                log.exception("상담 중 조건 탐색 실패")
                reply["content"] += f"\n\n⚠ {user_message(e)}"
                r = None
        if r is not None:
            # 결과 설명은 코드가 카드 내용으로 만든다 (근거 없는 추천 이유가 섞이지 않게)
            found = (f"조건 \"{turn.condition}\"으로 모델 메타데이터 후보 {r.candidates}건을 살펴, 조건을 모두 확인할 수 있는 "
                     f"{len(r.results)}개를 골랐습니다." if r.results else
                     f"조건 \"{turn.condition}\"을 모두 만족하는 모델을 메타데이터에서 찾지 못했습니다. 조건을 줄여 말씀해 주세요.")
            reply["content"] = f"{turn.message}\n\n{found}"
            reply["cards"] = r.results
            reply["notes"] += r.notes
    history.append(reply)
    render_consult_message(reply, len(history) - 1)


def render_consult_message(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        cards = msg.get("cards") or []
        if cards:
            cols = st.columns(2)
            for j, rec_item in enumerate(cards):
                with cols[j % 2]:
                    render_rec_card(rec_item, key=f"consult_pick_{idx}_{j}")
        for n in msg.get("notes", []):
            st.caption(f"ℹ {n}")


# 체험 토글은 사양 적용(st.rerun)처럼 그려지기 전에 끝나는 실행이 끼면 Streamlit이 값을 지워 저절로 꺼졌다.
# 매 실행 처음에 값을 다시 써 두면 위젯이 안 그려진 실행이 있어도 유지된다.
PERSISTENT_KEYS = ("speed_exp", "persona_exp")


def main() -> None:
    for k in PERSISTENT_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]
    st.html(CHAT_INPUT_CSS)
    st.title("Can I run this model?")
    st.caption("로컬 LLM 실행 가능성·양자화·라이선스를 모델 메타데이터와 모델 카드 근거로 답합니다.")
    models = preflight()
    if models is None:
        return
    m, docs_slot = sidebar(models)
    if current_hw() is None and not st.session_state.get("hw_skip"):
        onboarding()
    if st.session_state.get("mode") == MODE_CONSULT:
        consult_page()
        return
    recommend_cards()
    if m is None:
        st.info("왼쪽에서 질문할 모델을 선택하세요.")
        return

    model_header(m)
    hw = current_hw()
    report, speeds, calc = None, {}, ""
    if hw:
        report, speeds = fit_card(m, hw)
        calc = calc_text(m, hw, report, {q: s.tok_s for q, s in speeds.items()})
    else:
        calc = memory_text(m)  # 사양을 안 골라도 "Q4면 메모리 얼마나?"에는 계산값으로 답한다
        st.info("위에서 내 PC 사양을 고르거나 사이드바에서 하드웨어를 감지하면 이 모델의 실행 가능 판정이 표시됩니다.")

    # 체험 토글을 대화보다 먼저 그린다: 🎭 대화 체험이면 근거 Q&A와 다른 대화 기록·입력 안내를 쓴다
    stream_tok_s, persona_quant = experience_controls(m, report, speeds)
    persona_mode = persona_quant is not None
    key = f"{m['id']}::persona" if persona_mode else m["id"]
    history = chats().setdefault(key, [])  # 대화 기록은 모델별로, 근거 Q&A와 🎭 대화는 따로 (F2009)
    # 입력창은 화면 아래에 고정되므로 먼저 받아도 위치는 같다. 질문이 들어온 실행에서는 안내 말풍선을 그리지 않는다
    placeholder = (f"🎭 {m['name']}(연기)에게 아무거나 말 걸어 보세요" if persona_mode
                   else f"💬 {m['name']}에 대해 질문하세요 (예: Q4로 돌리면 메모리 얼마나 필요해?)")
    typed = st.chat_input(placeholder)
    question = typed or st.session_state.pop("pending_question", None)
    for msg in history:
        render_message(msg)
    if not history and not question:
        welcome(m, persona_mode)

    if question:
        prior = history_for_prompt(history)
        user_msg = {"role": "user", "content": question}
        history.append(user_msg)
        render_message(user_msg)
        if persona_mode:
            reply = persona_reply(m, persona_quant, question, prior)
        else:
            reply = handle_question(m, question, calc, prior)
        history.append(reply)
        render_message(reply, stream_tok_s=stream_tok_s)
    render_doc_status(m, docs_slot)


try:
    main()
except Exception:  # noqa: BLE001 - 예상 못 한 오류도 화면에는 안내만 (NF2004). rerun/stop은 BaseException이라 지나간다
    log.exception("화면 처리 중 오류")
    st.error("화면을 그리는 중 오류가 발생했습니다. 새로고침해도 계속되면 logs/app.log를 확인하세요.")
