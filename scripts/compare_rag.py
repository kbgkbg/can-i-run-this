"""RAG 사용 전후 답변 비교 (F2015).

실행: python scripts/compare_rag.py
대표 질문마다 같은 LLM으로 (1) RAG 없이 (2) 이 서비스의 RAG 파이프라인으로 답하게 해 docs/eval/rag_comparison.md에 나란히 저장한다.

평가 기준 (명세 4.6) — 자동 채점 규칙
- 사실 정확성: 정답 수치(models.json·모델 카드·기술 보고서·계산값에서 확정)가 답변에 모두 있으면 O, 일부면 △, 없으면 X
  (모르는 것을 묻는 질문은 해당 없음)
- 근거 제시: 답변에 실제 인용된 근거가 있으면 O (RAG 없이는 근거 목록 자체가 없다)
- 모르는 것을 모른다고 답함: 정답이 문서에 없는 질문에서 "확인되지 않/알 수 없/공개되지 않" 류로 답하고
  수치를 지어내지 않으면 O
자동 채점은 수치 포함 여부만 보는 규칙이라, 표 아래 원문 답변으로 사람이 다시 확인한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from app.config import PROJECT_ROOT  # noqa: E402
from app.data_prep.item_text import load_models  # noqa: E402
from app.features.fit import assess, calc_text  # noqa: E402
from app.features.hw_detect import manual  # noqa: E402
from app.features.speed import estimate_all  # noqa: E402
from app.rag.citations import _NUM, _norm_numbers  # noqa: E402
from app.rag.llm import chat_model  # noqa: E402
from app.rag.pipeline import ask  # noqa: E402

NO_RAG_SYSTEM = "너는 로컬 LLM 배포 전문가다. 사용자의 질문에 한국어로 간결하게 답한다."
UNKNOWN_PAT = re.compile(r"확인되지 않|알 수 없|공개되지 않|정보가 없|명시되어 있지 않|알려지지 않")

# (모델, 질문, 정답 수치, 정답 출처, 하드웨어(계산값이 필요한 질문만), 모르는 것을 묻는 질문인지)
CASES = [
    ("Qwen/Qwen3-8B", "Qwen3-8B의 레이어 수와 KV 헤드 수는?", ["36", "8"], "models.json (config.json)", None, False),
    ("Qwen/Qwen3-8B", "Qwen3-8B를 thinking 모드로 쓸 때 권장 샘플링 설정(Temperature, TopP, TopK)은?",
     ["0.6", "0.95", "20"], "Qwen3-8B 모델 카드 Best Practices", None, False),
    ("Qwen/Qwen3-8B", "Qwen3 모델들은 사전 학습에 토큰을 얼마나 썼어?", ["36"], "Qwen3 기술 보고서 p.3 (36 trillion)", None, False),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "이 모델의 레이어 수와 최대 컨텍스트 길이는?",
     ["52", "262144"], "models.json (config.json)", None, False),
    ("Qwen/Qwen3-8B", "RTX 3080(10GB)에서 Q4_K_M으로 컨텍스트 4096을 쓰면 메모리가 얼마나 필요해?",
     ["5.8"], "계산값 (fit.py)", ("RTX 3080", 32), False),
    ("Qwen/Qwen3-8B", "Qwen3-8B 학습에 GPU를 몇 시간 썼고 학습 비용은 얼마였어?", [], "문서에 없음", None, True),
    ("Qwen/Qwen3-8B", "이 모델을 파인튜닝해서 재배포할 때 지켜야 할 조건은?", [], "Qwen3-8B LICENSE (Apache 2.0) 원문", None, False),
]


def score_fact(answer: str, truth: list[str]) -> str:
    if not truth:
        return "-"
    # 답변에서 독립된 수치만 뽑아 비교한다 ("Qwen3-8B"의 8을 정답 8로 세지 않게)
    found = set(_NUM.findall(_norm_numbers(answer)))
    hit = sum(t in found for t in truth)
    return "O" if hit == len(truth) else ("△" if hit else "X")


def score_unknown(answer: str, is_unknown: bool) -> str:
    if not is_unknown:
        return "-"
    invented = re.search(r"\d[\d,.]*\s*(시간|GPU|달러|\$|원|억|만)", answer)
    return "O" if UNKNOWN_PAT.search(answer) and not invented else "X"


def tally(scores: list[tuple[str, str, str]], i: int) -> str:
    target = [s[i] for s in scores if s[i] != "-"]
    return f"{sum(x == 'O' for x in target)}/{len(target)}"


def main() -> int:
    models = {m["id"]: m for m in load_models()}
    rows, details, no_scores, rag_scores = [], [], [], []
    for model_id, q, truth, truth_src, hw_spec, is_unknown in CASES:
        m = models[model_id]
        no_rag = chat_model().invoke([SystemMessage(NO_RAG_SYSTEM), HumanMessage(f"[모델] {model_id}\n{q}")]).content

        calc = ""
        if hw_spec:
            hw = manual(*hw_spec)
            report = assess(m, hw)
            calc = calc_text(m, hw, report, {k: v.tok_s for k, v in estimate_all(m, hw, report).items()})
        r = ask(m, q, calc=calc)
        rag = r.answer if r.ok else f"(오류) {r.error}"

        s_no = (score_fact(no_rag, truth), "X", score_unknown(no_rag, is_unknown))
        s_rag = (score_fact(rag, truth), "O" if r.cited else "X", score_unknown(rag, is_unknown))
        no_scores.append(s_no)
        rag_scores.append(s_rag)
        rows.append(f"| {q} | {truth_src} | {' / '.join(s_no)} | {' / '.join(s_rag)} |")
        cited = ", ".join(f"[근거 {e.n}] {e.label} {e.title}" for e in r.cited) or "없음"
        details += [
            f"### {q}", f"- 모델: `{model_id}` · 정답 출처: {truth_src}" + (f" · 정답 수치: {', '.join(truth)}" if truth else ""), "",
            "**RAG 없이**", "", no_rag.strip(), "",
            "**RAG 사용**", "", rag.strip(), "", f"- 인용된 근거: {cited}",
            *(f"- ⚠ 근거에서 확인되지 않은 수치: {u}" for u in r.unverified), "",
        ]
        print(f"[완료] {q} | RAG 없이 {s_no} | RAG {s_rag}")

    out = PROJECT_ROOT / "docs" / "eval" / "rag_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        "# RAG 사용 전후 답변 비교 (F2015)", "",
        "같은 LLM(config.yaml llm.chat_model)에 같은 질문을 (1) 근거 없이 (2) 이 서비스의 RAG 파이프라인으로 물었다.",
        "채점은 `사실 정확성 / 근거 제시 / 모르는 것을 모른다고 답함` 순서 (O 충족, △ 일부, X 미충족, - 해당 없음).",
        "자동 채점 규칙은 `scripts/compare_rag.py` 머리말에 있다. 아래 원문 답변으로 직접 다시 확인할 것.", "",
        "| 질문 | 정답 출처 | RAG 없이 | RAG 사용 |", "|---|---|---|---|", *rows, "",
        "| 기준 | RAG 없이 | RAG 사용 |", "|---|---|---|",
        *(f"| {name} | {tally(no_scores, i)} | {tally(rag_scores, i)} |"
          for i, name in enumerate(["사실 정확성 (O 개수/대상)", "근거 제시", "모르는 것을 모른다고 답함"])),
        "", "## 질문별 답변 원문", "", *details,
    ]), encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
