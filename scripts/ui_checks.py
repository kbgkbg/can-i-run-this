"""화면 기능 점검 (F2009~F2012) — Streamlit AppTest로 실제 앱 스크립트를 실행해 확인한다.

실행: python scripts/ui_checks.py
비정상 상황(키 누락, 색인 0건, 잘못된 키)은 환경을 바꿔야 해서 케이스마다 별도 프로세스로 돌린다.
결과는 docs/eval/ui_test_results.md에 저장한다. 화면 캡처는 docs/screenshots/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "ui" / "streamlit_app.py")

CASE_CODE = r'''
import json, os, sys, tempfile
sys.path.insert(0, ROOT)
case = sys.argv[1]
if case == "nokey":
    os.environ["OPENAI_API_KEY"] = ""
if case == "wrongkey":
    os.environ["OPENAI_API_KEY"] = "invalid-key-for-test"
from app.config import get_config
if case == "noindex":
    get_config()["paths"]["chroma_dir"] = tempfile.mkdtemp()
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(APP, default_timeout=240).run()
out = {"error": [e.value for e in at.error], "warning": [w.value for w in at.warning], "exception": len(at.exception)}
if case in ("normal", "wrongkey"):
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
    at.chat_input[0].set_value("Qwen3-8B의 권장 샘플링 설정은?").run()
    out["status"] = [s.label for s in at.status]
    out["messages_8b"] = len(at.chat_message)
    out["assistant"] = [m.value[:120] for m in at.chat_message[-1].markdown][:1]
    out["cited"] = [m.value for m in at.chat_message[-1].markdown if m.value.startswith("**[근거")]
    out["exception"] = len(at.exception)
    if case == "normal":
        at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-4B").run()
        out["messages_4b"] = [c.name for c in at.chat_message]
        at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
        out["messages_8b_back"] = len(at.chat_message)
if case == "ctx":
    # 회귀: 컨텍스트를 32k로 올린 뒤 최대 컨텍스트가 작은 모델(TinyLlama 2k)로 바꾸면 위젯 값이 최댓값을 넘어 예외가 났다
    from app.data_prep.hw_tables import by_name
    at.sidebar.selectbox(key="side_nv_gpu").set_value(by_name("RTX 3080")).run()
    at.sidebar.button(key="side_nv_apply").click().run()
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
    at.number_input(key="context::Qwen/Qwen3-8B").set_value(32768).run()
    at.sidebar.selectbox(key="model_id").select("TinyLlama/TinyLlama-1.1B-Chat-v1.0").run()
    out["small_ctx"] = at.number_input(key="context::TinyLlama/TinyLlama-1.1B-Chat-v1.0").value
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
    out["restored_ctx"] = at.number_input(key="context::Qwen/Qwen3-8B").value
    out["exception"] = len(at.exception)
if case == "persona":
    # 🎭 대화 체험: 문서 검색 없이 아무 주제나 대화, 이전 대화를 기억, 근거 Q&A와 기록 분리
    # 대형 체급(연출 실수 0개)으로 확인한다. 중형 이하는 의도적으로 틀리므로 기억 확인에 쓸 수 없다
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-32B").run()
    at.toggle(key="persona_exp").set_value(True).run()
    out["welcome"] = [x.name for x in at.chat_message]
    out["notice"] = any("연기(시뮬레이션)" in c.value for c in at.caption)
    at.chat_input[0].set_value("능금이 뭐야?").run()
    out["status"] = len(at.status)  # RAG 진행 상태(문서 수집)가 없어야 한다
    at.chat_input[0].set_value("방금 내가 무슨 단어 물어봤지?").run()
    out["names"] = [x.name for x in at.chat_message]
    pm = at.chat_message[-1]
    out["answer1"] = [x.value[:100] for x in at.chat_message[1].markdown][1:2]
    out["answer2"] = [x.value[:100] for x in pm.markdown][1:2]
    at.toggle(key="persona_exp").set_value(False).run()
    out["rag_names"] = [x.name for x in at.chat_message]
    out["exception"] = len(at.exception)
if case == "speed":
    # 사양을 건너뛰고 모델만 직접 고른 경우에도 속도 체험이 되는지 + 대역폭 TODO GPU에 직접 입력
    from app.data_prep.hw_tables import by_name
    at.button(key="onb_skip").click().run()
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-0.6B").run()
    at.toggle(key="speed_exp").set_value(True).run()
    out["detected"] = (at.session_state["hw"] or {}).get("method")
    at.chat_input[0].set_value("Qwen3-0.6B 레이어 수는?").run()
    out["play_detected"] = [c.value for c in at.caption if "재생 중" in c.value][:1]
    # 예전에 대역폭 TODO였던 RTX 4070: 이제 사양표 값(504 GB/s)으로 바로 재생된다 (입력칸 없음)
    at.sidebar.selectbox(key="side_nv_gpu").set_value(by_name("RTX 4070")).run()
    out["bw_input"] = any(n.key == "side_nv_bw" for n in at.sidebar.number_input)
    at.sidebar.button(key="side_nv_apply").click().run()
    hw = at.session_state["hw"] or {}
    out["manual_bw"] = [hw.get("spec_name"), hw.get("bandwidth_gbps"), hw.get("bandwidth_source")]
    at.chat_input[0].set_value("Qwen3-0.6B 컨텍스트 길이는?").run()
    out["play_manual"] = [c.value for c in at.caption if "재생 중" in c.value][-1:]
    out["exception"] = len(at.exception)
if case == "consult":
    at.sidebar.radio(key="mode").set_value("추천 상담").run()
    at.chat_input[0].set_value("모델 추천해줘").run()
    at.chat_input[0].set_value("코딩용이면 돼. 그래픽카드는 3080, 램 32기가").run()
    last = at.chat_message[-1]
    out["cards"] = [b.key for b in last.button]
    out["hw"] = (at.session_state["hw"] or {}).get("spec_name")
    if last.button:
        last.button[0].click().run()
        out["mode_after"] = at.session_state["mode"]
        out["model_after"] = at.session_state["model_id"]
    out["exception"] = len(at.exception)
if case == "onboarding":
    from app.data_prep.hw_tables import by_name
    out["onb_shown"] = any("먼저 내 PC 사양" in h.value for h in at.subheader)
    at.selectbox(key="onb_mac_chip").set_value(by_name("Apple M3 Max (40-core GPU)")).run()
    at.selectbox(key="onb_mac_mem").set_value(36).run()
    at.button(key="onb_mac_apply").click().run()
    hw = at.session_state["hw"] or {}
    out["hw"] = [hw.get("kind"), hw.get("spec_name"), hw.get("gpu_mem_gb"), hw.get("bandwidth_gbps")]
    out["onb_after"] = any("먼저 내 PC 사양" in h.value for h in at.subheader)
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-30B-A3B").run()
    out["summary"] = [m.value for m in at.markdown if "(계산값)" in m.value and "**" in m.value][:1]
    out["ctx_table"] = any("컨텍스트 길이별 차이" in e.label for e in at.expander)
    out["ctx_rows"] = len(at.dataframe[1].value) if len(at.dataframe) > 1 else 0
    out["exception"] = len(at.exception)
if case == "welcome":
    # 아직 문서를 수집하지 않은 모델: 안내 말풍선·첫 질문 안내가 보이고, 예시 질문 버튼으로 질문이 처리된다
    from app.collect import cache
    from app.data_prep.item_text import load_models
    # 작은 모델부터 아직 문서를 수집하지 않은 것 (테스트를 돌릴 때마다 하나씩 수집되므로 고정 ID를 쓰지 않는다)
    mid = next(m["id"] for m in sorted(load_models(), key=lambda m: m["params"] or 0)
               if (m["params"] or 0) > 3e8 and not cache.indexed_docs(m["id"]))
    at.sidebar.selectbox(key="model_id").select(mid).run()
    out["model"] = mid
    out["welcome"] = [c.name for c in at.chat_message]
    out["first_notice"] = any("처음 질문하면" in c.value for c in at.caption)
    at.button(key="example::권장 샘플링 설정은?").click().run()
    out["status"] = [s.label for s in at.status]
    out["after"] = [c.name for c in at.chat_message]
    out["question"] = [mk.value for c in at.chat_message if c.name == "user" for mk in c.markdown][:1]
    out["exception"] = len(at.exception)
print("@@" + json.dumps(out, ensure_ascii=False))
'''


def run_case(case: str) -> dict:
    code = f"ROOT={str(ROOT)!r}\nAPP={APP!r}\n" + CASE_CODE
    p = subprocess.run([sys.executable, "-c", code, case], cwd=ROOT, capture_output=True, text=True, timeout=600)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("@@")), None)
    return json.loads(line[2:]) if line else {"crash": p.stderr[-500:]}


def main() -> int:
    rows = []
    n = run_case("normal")
    rows.append(("F2009", "모델별 대화 분리: 8B에서 질문 → 4B 전환 → 8B 복귀",
                 n.get("messages_8b") == 2 and n.get("messages_4b") == ["assistant"] and n.get("messages_8b_back") == 2,
                 f"8B 메시지 {n.get('messages_8b')} → 4B {n.get('messages_4b')} (안내 말풍선만) → 8B {n.get('messages_8b_back')}"))
    rows.append(("F2008", "답변 아래에 실제 인용된 근거만 원문 링크와 표시", bool(n.get("cited")),
                 "; ".join(n.get("cited", []))[:150]))
    rows.append(("F2012", "진행 상태 표시 후 완료로 바뀜", "답변 완료" in n.get("status", []), f"status={n.get('status')}"))
    k = run_case("nokey")
    rows.append(("F2012", "API 키 누락 안내", any("OPENAI_API_KEY" in e for e in k.get("error", [])), (k.get("error") or [""])[0][:80]))
    i = run_case("noindex")
    rows.append(("F2012", "색인 0건이면 색인 스크립트 실행 안내", any("build_meta_index" in w for w in i.get("warning", [])),
                 (i.get("warning") or [""])[0][:80]))
    w = run_case("wrongkey")
    rows.append(("NF2004", "잘못된 키: 사용자 문구만 표시, 화면 예외 없음",
                 w.get("exception") == 0 and any("API 키" in a for a in w.get("assistant", [])), (w.get("assistant") or [""])[0][:80]))

    c = run_case("ctx")
    rows.append(("F2010", "컨텍스트 32k 설정 후 최대 2k 모델로 전환해도 예외 없음, 돌아오면 값 복원",
                 c.get("exception") == 0 and c.get("small_ctx") == 2048 and c.get("restored_ctx") == 32768,
                 f"전환 후 {c.get('small_ctx')}, 복귀 후 {c.get('restored_ctx')}, 예외 {c.get('exception')}"))
    pc = run_case("persona")
    rows.append(("F2010", "🎭 대화 체험: 문서 검색 없이 아무 주제나 답하고 이전 대화를 기억, 근거 Q&A 기록과 분리",
                 pc.get("exception") == 0 and pc.get("welcome") == ["🎭"] and pc.get("notice") and pc.get("status") == 0
                 and pc.get("names") == ["user", "🎭", "user", "🎭"] and "능금" in "".join(pc.get("answer2", []))
                 and "자료" not in "".join(pc.get("answer1", [])) and pc.get("rag_names") == ["assistant"],
                 f"안내 {pc.get('welcome')}, 문서 수집 상태 {pc.get('status')}개, 말풍선 {pc.get('names')}, "
                 f"답1 {pc.get('answer1')}, 후속 답 {pc.get('answer2')}, 끄면 {pc.get('rag_names')}"))
    sp = run_case("speed")
    rows.append(("F2010", "속도 체험: 사양을 건너뛰고 모델만 골라도 켜면 자동 감지 후 재생, 예전 TODO였던 RTX 4070은 사양표 대역폭으로 재생",
                 sp.get("exception") == 0 and sp.get("detected") != "manual" and bool(sp.get("play_detected"))
                 and sp.get("manual_bw", [None])[:2] == ["RTX 4070", 504.0] and sp.get("bw_input") is False and bool(sp.get("play_manual")),
                 f"자동 감지({sp.get('detected')}) 후 {sp.get('play_detected')} / 직접 입력 {sp.get('manual_bw')} 후 {sp.get('play_manual')}"))
    wl = run_case("welcome")
    rows.append(("F2009", "빈 대화: 안내 말풍선과 첫 질문 소요 안내, 예시 질문 버튼으로 바로 질문",
                 wl.get("exception") == 0 and wl.get("welcome") == ["assistant"] and wl.get("first_notice")
                 and wl.get("after") == ["user", "assistant"] and any("첫 질문" in x for x in wl.get("status", [])),
                 f"{wl.get('model')}: 질문 전 {wl.get('welcome')}, 첫 질문 안내={wl.get('first_notice')}, "
                 f"진행 상태 {str(wl.get('status'))[:60]} → 질문 {wl.get('question')} 후 {wl.get('after')}"))
    cs = run_case("consult")
    rows.append(("F2011", "추천 상담: 되묻기 1회 후 검색, 대화 속 하드웨어 적용, 카드 클릭 시 대화로 전환",
                 cs.get("exception") == 0 and bool(cs.get("cards")) and cs.get("hw") == "RTX 3080"
                 and cs.get("mode_after") == "모델에게 묻기" and bool(cs.get("model_after")),
                 f"카드 {len(cs.get('cards', []))}개, 하드웨어 {cs.get('hw')}, 전환 후 {cs.get('mode_after')} / {cs.get('model_after')}"))
    ob = run_case("onboarding")
    rows.append(("F2010", "첫 화면 사양 선택(Mac 탭): M3 Max 40코어·36GB → 통합 메모리 판정, 컨텍스트별 차이 표",
                 ob.get("exception") == 0 and ob.get("onb_shown") and not ob.get("onb_after")
                 and ob.get("hw", [None])[0] == "apple" and ob.get("ctx_table") and ob.get("ctx_rows", 0) >= 3,
                 f"첫 화면 {ob.get('onb_shown')} → 적용 후 {ob.get('onb_after')}, 하드웨어 {ob.get('hw')}, "
                 f"판정 {str(ob.get('summary'))[:60]}, 컨텍스트 표 {ob.get('ctx_rows')}행"))
    passed = sum(ok for _, _, ok, _ in rows)
    lines = ["# 화면 기능 점검 결과 (Streamlit AppTest)", "",
             f"- 실행: `python scripts/ui_checks.py` · 통과 {passed}/{len(rows)}", "",
             "| 요구사항 | 점검 내용 | 결과 | 실행 결과 |", "|---|---|---|---|",
             *(f"| {r} | {name} | {'PASS' if ok else 'FAIL'} | {' '.join(str(d).split()).replace('|', '/')} |" for r, name, ok, d in rows)]
    out = ROOT / "docs" / "eval" / "ui_test_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r, name, ok, d in rows:
        print("PASS" if ok else "FAIL", r, name, "|", d)
    print(f"통과 {passed}/{len(rows)} -> {out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
