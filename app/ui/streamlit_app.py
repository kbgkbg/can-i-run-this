"""Streamlit UI (F2009, F2012).

실행: streamlit run app/ui/streamlit_app.py
- 사이드바: 모델 선택, 색인 현황(메타데이터 건수, 선택 모델의 수집 문서)
- 메인: 선택 모델 헤더 + 모델별로 분리된 대화 + 진행 상태 + 인용된 근거
화면에는 안내 문구만 띄우고 프롬프트 전문·스택 트레이스는 보이지 않는다 (NF2004).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # streamlit run은 스크립트 폴더만 경로에 넣는다

import streamlit as st  # noqa: E402

from app.collect import cache  # noqa: E402
from app.config import missing_required_env  # noqa: E402
from app.data_prep.item_text import format_params, load_models  # noqa: E402
from app.index.indexer import count_model_docs  # noqa: E402
from app.index.store import META_COLLECTION, count  # noqa: E402
from app.rag.pipeline import ask  # noqa: E402

st.set_page_config(page_title="Can I run this model?", page_icon="🖥️", layout="wide")


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
def sidebar(models: dict[str, dict]) -> dict | None:
    with st.sidebar:
        st.header("모델 선택")
        ids = list(models)  # models.json 순서 = 다운로드순
        model_id = st.selectbox(
            "질문할 모델", ids, index=None, key="model_id",
            format_func=lambda i: f"{i}  ·  {short_params(models[i].get('params'))}",
            placeholder="모델 이름을 입력해 검색하세요",
        )
        st.divider()
        st.caption(f"모델 메타데이터 색인: {count(META_COLLECTION)}건")
        if model_id:
            m = models[model_id]
            docs = cache.indexed_docs(model_id)
            st.caption(f"수집된 문서 {len(docs)}건 · 청크 {count_model_docs(model_id)}개")
            for d in docs.values():
                st.caption(f"· [{md_escape(d['title'])}]({d['source_url']})")
            if m.get("config_status") != "ok":
                st.caption("⚠ 이 모델은 gated 저장소라 레이어 수 등 일부 메타데이터가 없습니다.")
        return models.get(model_id) if model_id else None


# ------------------------------------------------------------------ 메인
def model_header(m: dict) -> None:
    st.subheader(m["id"])
    cols = st.columns(4)
    cols[0].metric("파라미터", short_params(m.get("params")))
    cols[1].metric("컨텍스트", f"{m['max_context']:,}" if m.get("max_context") else "정보 없음")
    cols[2].metric("레이어", m.get("num_layers") or "정보 없음")
    cols[3].metric("라이선스", m.get("license") or "정보 없음")
    st.caption(f"[Hugging Face 모델 페이지]({m['hf_url']})")


def render_sources(msg: dict) -> None:
    cited = msg.get("cited", [])
    if not cited:
        st.caption(msg.get("notice") or "이 답변에는 인용된 근거가 없습니다.")
        return
    with st.expander(f"답변에 사용한 근거 {len(cited)}건"):
        for e in cited:
            st.markdown(f"**[근거 {e['n']}]** {md_escape(e['title'])} · {e['label']} · [원문]({e['url']})")


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and not msg.get("error"):
            render_sources(msg)
            for w in msg.get("warnings", []):
                st.caption(f"ℹ {w}")
            if msg.get("elapsed"):
                st.caption(f"처리 시간 {msg['elapsed']:.1f}초")


def handle_question(m: dict, question: str) -> dict:
    with st.status("질문을 처리하는 중...", expanded=True) as status:
        result = ask(m, question, on_step=lambda stage, text: st.write(f"**{stage}** · {text}"))
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
        "warnings": result.warnings,
        "elapsed": sum(result.timings.values()),
    }


def main() -> None:
    st.title("Can I run this model?")
    st.caption("로컬 LLM 실행 가능성·양자화·라이선스를 모델 메타데이터와 모델 카드 근거로 답합니다.")
    models = preflight()
    if models is None:
        return
    m = sidebar(models)
    if m is None:
        st.info("왼쪽에서 질문할 모델을 선택하세요.")
        return

    model_header(m)
    history = chats().setdefault(m["id"], [])  # 대화 기록은 모델별로 분리 (F2009)
    for msg in history:
        render_message(msg)

    question = st.chat_input(f"{m['name']}에 대해 질문하세요 (예: Q4로 돌리면 메모리 얼마나 필요해?)")
    if question:
        user_msg = {"role": "user", "content": question}
        history.append(user_msg)
        render_message(user_msg)
        reply = handle_question(m, question)
        history.append(reply)
        render_message(reply)


main()
