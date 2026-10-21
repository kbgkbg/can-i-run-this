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
        out["messages_4b"] = len(at.chat_message)
        at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
        out["messages_8b_back"] = len(at.chat_message)
if case == "ctx":
    # 회귀: 컨텍스트를 32k로 올린 뒤 최대 4k 모델로 바꾸면 위젯 값이 최댓값을 넘어 예외가 났다
    at.sidebar.selectbox(key="manual_gpu").select("RTX 3080").run()
    next(b for b in at.sidebar.button if b.label == "적용").click().run()
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
    at.number_input(key="context::Qwen/Qwen3-8B").set_value(32768).run()
    at.sidebar.selectbox(key="model_id").select("utter-project/EuroLLM-1.7B-Instruct").run()
    out["small_ctx"] = at.number_input(key="context::utter-project/EuroLLM-1.7B-Instruct").value
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-8B").run()
    out["restored_ctx"] = at.number_input(key="context::Qwen/Qwen3-8B").value
    out["exception"] = len(at.exception)
if case == "persona":
    at.sidebar.selectbox(key="model_id").select("Qwen/Qwen3-0.6B").run()
    at.toggle(key="persona_exp").set_value(True).run()
    at.chat_input[0].set_value("Qwen3-0.6B의 레이어 수와 컨텍스트 길이는?").run()
    pm = at.chat_message[-1]
    out["names"] = [x.name for x in at.chat_message]
    out["notice"] = any("연기(시뮬레이션)" in c.value for c in pm.caption)
    out["changes"] = [x.value[:120] for x in pm.expander[0].markdown] if pm.expander else []
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
                 n.get("messages_8b") == 2 and n.get("messages_4b") == 0 and n.get("messages_8b_back") == 2,
                 f"8B 메시지 {n.get('messages_8b')} → 4B {n.get('messages_4b')} → 8B {n.get('messages_8b_back')}"))
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
    rows.append(("F2010", "컨텍스트 32k 설정 후 4k 모델로 전환해도 예외 없음, 돌아오면 값 복원",
                 c.get("exception") == 0 and c.get("small_ctx") == 4096 and c.get("restored_ctx") == 32768,
                 f"전환 후 {c.get('small_ctx')}, 복귀 후 {c.get('restored_ctx')}, 예외 {c.get('exception')}"))
    pc = run_case("persona")
    rows.append(("F2010", "🎭 흉내 답변: 근거 답변과 별도 말풍선, 연기 안내, 바뀐 수치 비교 표시",
                 pc.get("exception") == 0 and pc.get("names", [])[-1:] == ["🎭"] and pc.get("notice") and bool(pc.get("changes")),
                 f"말풍선 {pc.get('names')}, 안내={pc.get('notice')}, 변경={str(pc.get('changes'))[:90]}"))
    cs = run_case("consult")
    rows.append(("F2011", "추천 상담: 되묻기 1회 후 검색, 대화 속 하드웨어 적용, 카드 클릭 시 대화로 전환",
                 cs.get("exception") == 0 and bool(cs.get("cards")) and cs.get("hw") == "RTX 3080"
                 and cs.get("mode_after") == "모델에게 묻기" and bool(cs.get("model_after")),
                 f"카드 {len(cs.get('cards', []))}개, 하드웨어 {cs.get('hw')}, 전환 후 {cs.get('mode_after')} / {cs.get('model_after')}"))
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
