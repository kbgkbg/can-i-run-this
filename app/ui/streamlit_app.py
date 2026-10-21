"""Streamlit UI (F2009, F2010, F2011, F2012).

실행: streamlit run app/ui/streamlit_app.py
- 사이드바: 모델 선택, 조건으로 모델 찾기, 하드웨어 감지/직접 입력, 색인 현황(메타데이터 건수, 선택 모델의 수집 문서)
- 메인: 조건 탐색 결과 카드 + 선택 모델 헤더 + 실행 가능 판정 카드 + 모델별로 분리된 대화 + 진행 상태 + 인용된 근거
화면에는 안내 문구만 띄우고 프롬프트 전문·스택 트레이스는 보이지 않는다 (NF2004).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # streamlit run은 스크립트 폴더만 경로에 넣는다

import psutil  # noqa: E402
import streamlit as st  # noqa: E402

from app.collect import cache  # noqa: E402
from app.config import get_config, missing_required_env  # noqa: E402
from app.data_prep.hw_tables import load_specs  # noqa: E402
from app.data_prep.item_text import format_params, load_models  # noqa: E402
from app.features import hw_detect  # noqa: E402
from app.features.fit import FitReport, assess, calc_text  # noqa: E402
from app.features.hw_detect import Hardware  # noqa: E402
from app.features.recommend import search as recommend_search  # noqa: E402
from app.features.speed import SpeedEstimate, estimate_all  # noqa: E402
from app.index.indexer import count_model_docs  # noqa: E402
from app.index.store import META_COLLECTION, count  # noqa: E402
from app.log import get_logger  # noqa: E402
from app.rag.pipeline import ask  # noqa: E402

log = get_logger("ui")
st.set_page_config(page_title="Can I run this model?", page_icon="🖥️", layout="wide")

NO_GPU = "GPU 없음 (CPU만)"
VERDICT_ICON = {"GPU에 전부 올라감": "🟢", "일부 CPU 오프로딩 필요": "🟡", "CPU·RAM만으로 가능(느림)": "🟠", "불가": "🔴", "정보 부족": "⚪"}


@st.cache_data(show_spinner=False)
def models_by_id() -> dict[str, dict]:
    return {m["id"]: m for m in load_models()}


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
    return models


# ------------------------------------------------------------------ 사이드바
def hardware_panel() -> None:
    st.header("내 하드웨어")
    if st.button("하드웨어 감지", width="stretch"):
        try:
            st.session_state["hw"] = hw_detect.detect().to_dict()
        except Exception:  # noqa: BLE001
            log.exception("하드웨어 감지 실패")
            st.warning("하드웨어를 자동으로 감지하지 못했습니다. 아래에서 직접 입력하세요.")

    with st.expander("직접 입력"):
        names = [NO_GPU] + [s.name for s in load_specs()]
        gpu = st.selectbox("GPU / 칩", names, key="manual_gpu")
        default_ram = round(psutil.virtual_memory().total / 1024 ** 3)
        ram = st.number_input("RAM (GB)", min_value=1, max_value=2048, value=int(default_ram), key="manual_ram")
        if st.button("적용", width="stretch"):
            st.session_state["hw"] = hw_detect.manual(None if gpu == NO_GPU else gpu, ram).to_dict()

    hw = current_hw()
    if hw is None:
        st.caption("하드웨어를 감지하거나 입력하면 실행 가능 판정이 표시됩니다.")
        return
    if hw.kind == "apple":
        st.caption(f"**{hw.gpu_name}** · 통합 메모리 {hw.ram_gb:.0f} GB (GPU 사용 {hw.gpu_mem_gb:.1f} GB)")
    elif hw.gpu_name:
        st.caption(f"**{hw.gpu_name}** · VRAM {hw.vram_gb:.1f} GB · RAM {hw.ram_gb:.0f} GB")
    else:
        st.caption(f"GPU 없음 · RAM {hw.ram_gb:.0f} GB · {hw.cpu_name}")
    if hw.bandwidth_gbps:
        st.caption(f"메모리 대역폭 {hw.bandwidth_gbps:g} GB/s · [사양 출처]({hw.bandwidth_source})")
    st.caption(f"감지 방법: {hw.method}")
    for n in hw.notes:
        st.caption(f"ℹ {n}")


def sidebar(models: dict[str, dict]) -> tuple[dict | None, Any]:
    """(선택 모델, 수집 문서 목록을 나중에 채울 사이드바 자리)"""
    with st.sidebar:
        st.header("모델 선택")
        ids = list(models)  # models.json 순서 = 다운로드순
        model_id = st.selectbox(
            "질문할 모델", ids, index=None, key="model_id",
            format_func=lambda i: f"{i}  ·  {short_params(models[i].get('params'))}",
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
            with cols[i % 2].container(border=True):
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
                st.button("이 모델로 대화", key=f"pick_{rec_item.model_id}", on_click=_choose_model,
                          args=(rec_item.model_id,), width="stretch")
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
        best = report.best()
        if report.missing:
            hint = " (gated 모델은 .env에 HF_TOKEN을 넣고 메타데이터를 다시 수집하면 채워집니다)" if m.get("gated") else ""
            st.markdown(f"⚪ **정보 부족** — {', '.join(report.missing)}이(가) 메타데이터에 없어 판정할 수 없습니다{hint}.")
        elif best is None:
            st.markdown("🔴 **불가** — 현재 메모리로는 어떤 양자화로도 올라가지 않습니다 (계산값).")
        else:
            sp = speeds[best.quant].tok_s
            st.markdown(f"{VERDICT_ICON[best.verdict]} **{best.quant}: {best.verdict}** — 필요 {best.total_gb:.1f} GB"
                        + (f", 추정 약 {sp:.0f} tok/s" if sp else "") + " (계산값)")

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
        with st.expander("판정 근거와 계산식"):
            st.markdown(
                f"- 파라미터 수 {format_params(m.get('params'))}, 레이어 {m.get('num_layers') or '정보 없음'}, "
                f"KV 헤드 {m.get('num_kv_heads') or '정보 없음'} — **[근거 1] (모델 메타데이터)** · [원문]({m['hf_url']})\n"
                + (f"- 메모리 대역폭 {hw.bandwidth_gbps:g} GB/s — **하드웨어 사양표** · [출처]({hw.bandwidth_source})\n"
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


def render_message(msg: dict, stream_tok_s: float | None = None) -> None:
    with st.chat_message(msg["role"]):
        if stream_tok_s and msg["role"] == "assistant" and not msg.get("error"):
            st.caption(f"⏱ 내 PC 추정 속도 {stream_tok_s:.1f} tok/s로 재생 중 (추정치, 실제와 다를 수 있음)")
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


def handle_question(m: dict, question: str, calc: str) -> dict:
    with st.status("질문을 처리하는 중...", expanded=True) as status:
        result = ask(m, question, calc=calc, on_step=lambda stage, text: st.write(f"**{stage}** · {text}"))
        c = result.collect
        if c is not None:
            if c.cache_hit:
                st.write(f"이미 수집된 문서 {c.docs_total}건 사용")
            elif c.fetched:
                st.write("새로 수집한 문서: " + ", ".join(d.title for d in c.fetched))
        if result.ok:
            status.update(label="답변 완료", state="complete", expanded=False)
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


def speed_controls(report: FitReport | None, speeds: dict[str, SpeedEstimate]) -> float | None:
    """'내 PC 속도로 체험' 토글. 켜면 선택한 양자화의 추정 속도를 돌려준다."""
    usable = [r.quant for r in (report.rows if report else []) if speeds.get(r.quant) and speeds[r.quant].tok_s]
    if not usable:
        return None
    c1, c2 = st.columns([1, 1])
    on = c1.toggle("내 PC 속도로 체험", key="speed_exp", help="답변을 이 PC에서 모델을 돌렸을 때의 추정 생성 속도로 보여 줍니다.")
    best = report.best()
    default = usable.index(best.quant) if best and best.quant in usable else 0
    quant = c2.selectbox("체험 양자화", usable, index=default, key="speed_quant", label_visibility="collapsed",
                         disabled=not on)
    return speeds[quant].tok_s if on else None


def main() -> None:
    st.title("Can I run this model?")
    st.caption("로컬 LLM 실행 가능성·양자화·라이선스를 모델 메타데이터와 모델 카드 근거로 답합니다.")
    models = preflight()
    if models is None:
        return
    m, docs_slot = sidebar(models)
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
        st.info("사이드바에서 하드웨어를 감지하거나 직접 입력하면 이 모델의 실행 가능 판정이 표시됩니다.")

    history = chats().setdefault(m["id"], [])  # 대화 기록은 모델별로 분리 (F2009)
    for msg in history:
        render_message(msg)

    stream_tok_s = speed_controls(report, speeds)
    question = st.chat_input(f"{m['name']}에 대해 질문하세요 (예: Q4로 돌리면 메모리 얼마나 필요해?)")
    if question:
        user_msg = {"role": "user", "content": question}
        history.append(user_msg)
        render_message(user_msg)
        reply = handle_question(m, question, calc)
        history.append(reply)
        render_message(reply, stream_tok_s=stream_tok_s)
    render_doc_status(m, docs_slot)


try:
    main()
except Exception:  # noqa: BLE001 - 예상 못 한 오류도 화면에는 안내만 (NF2004). rerun/stop은 BaseException이라 지나간다
    log.exception("화면 처리 중 오류")
    st.error("화면을 그리는 중 오류가 발생했습니다. 새로고침해도 계속되면 logs/app.log를 확인하세요.")
