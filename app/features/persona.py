"""🎭 이 모델과 대화해 보기 (체험용 연기).

고른 오픈 모델을 실제로 돌리지 않고(호스팅 추론 비용 없이), 같은 LLM(GMS)으로 그 모델 체급의 대화 상대를 연기한다.
목적은 "이 정도 지능이면 쓸 만하겠다"를 직접 말을 걸어 보며 체감하는 것이라, 문서 검색(RAG) 없이 아무 주제나 대화한다.
1단계: 기준 답. 범용 챗봇으로서 이전 대화를 이어 자기 지식으로 답한다 (상식·코딩·계산·글쓰기 등).
       처음에는 RAG 답변을 기준 답으로 썼는데, 그러면 문서에 없는 말("능금이 뭐야?")에는 "자료에 없어요"만 나와
       모델의 지능이 아니라 문서 검색 결과를 연기하게 됐다 (사용자 피드백).
2단계: 체급에 맞게 고쳐 쓴다. "작은 모델처럼 틀려라"는 지시는 잘 지켜지지 않아(0.6B 역할도 계산을 다 맞힘),
       넣을 사실·계산 오류 개수, 조건 누락, 반복, 문장 수를 정확히 지정하고 넣은 실수 목록을 함께 받는다.
체급(유효 파라미터 구간)과 실수 개수는 가정값이다 (config.yaml persona). 결과는 실제 모델 출력이 아니다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import get_config
from app.features.speed import active_params
from app.rag.citations import _NUM, _norm_numbers
from app.rag.llm import invoke_json, named_chat_model

PERSONA_NOTICE = "🎭 연기(시뮬레이션)입니다. 실제 이 모델의 출력이 아니며, 내용의 사실 여부를 믿지 마세요."

BASE_SYSTEM = ("너는 사용자 PC에서 돌아가는 범용 로컬 LLM 챗봇이다. 인터넷 검색 없이 자기 지식으로 무엇이든 대화한다 "
               "(상식, 코딩, 계산, 글쓰기, 잡담). 이전 대화를 이어서 자연스럽게 한국어로 답한다.")

DEGRADE_SYSTEM = """너는 LLM 체험 서비스의 연출 담당이다. 주어진 답변을 "{tier} 체급(유효 파라미터 약 {params_b}B) 로컬 모델이 썼을 법한 답"으로 고쳐 쓴다.

정확히 지킬 것
- 이 체급의 말투·능력: {ability}
{errors_rule}{drop}{repeat}{length}{quant}
- 실수는 우스꽝스럽지 않게, 그 크기 모델이 실제로 낼 법한 형태로 넣는다. 연출이라는 사실은 답변에 쓰지 않는다.

출력은 JSON 하나만: {{"answer": "고쳐 쓴 답변", "mistakes": ["넣은 실수를 한 줄씩 (원래 값 -> 바꾼 값)"]}}"""


@dataclass
class Persona:
    model_id: str
    quant: str
    params_b: float          # 유효 파라미터 (십억)
    tier: str
    rule: dict[str, Any]
    quant_note: str = ""


@dataclass
class PersonaAnswer:
    persona: Persona
    answer: str
    mistakes: list[str] = field(default_factory=list)   # LLM이 설명한 연출 실수 (빠뜨리거나 틀릴 수 있다)
    removed_numbers: list[str] = field(default_factory=list)  # 기준 답에 있었는데 사라진 수치 (코드로 비교)
    added_numbers: list[str] = field(default_factory=list)    # 흉내 답에 새로 생긴 수치 (코드로 비교)


def effective_params_b(model: dict[str, Any]) -> float:
    total = model["params"]
    if model.get("num_experts"):
        active, _ = active_params(model)
        return math.sqrt(total * active) / 1e9
    return total / 1e9


def make_persona(model: dict[str, Any], quant: str) -> Persona:
    cfg = get_config()["persona"]
    eff = effective_params_b(model)
    rule = next(t for t in cfg["tiers"] if eff <= t["max_b"])
    return Persona(model["id"], quant, round(eff, 1), rule["label"], rule, cfg["quant_notes"].get(quant, ""))


def _degrade_messages(p: Persona, question: str, base: str, history: list[tuple[str, str]] | None = None) -> list:
    r = p.rule
    system = DEGRADE_SYSTEM.format(
        tier=p.tier, params_b=p.params_b, ability=r.get("ability", ""),
        # 실수 0개 체급(대형 이상)에는 "의도를 엇나가게" 규칙을 주지 않는다. 함께 주면 대형도 앞 대화를 틀리게 기억했다
        errors_rule=(f"- 사실·수치·계산·코드 중 정확히 {r['errors']}곳을 그럴듯하게 틀린 내용으로 바꾼다 (그만큼 없으면 있는 만큼).\n"
                     "  틀릴 곳이 없는 잡담이면 대신 이 체급의 이해력만큼 질문 의도를 살짝 엇나가게 답한다. 틀린 곳을 표시하거나 암시하지 않는다.\n"
                     if r["errors"] else "- 사실·수치·계산·기억은 원래 답변 그대로 둔다. 내용을 바꾸지 말고 말투만 다듬는다.\n"),
        drop="- 질문 속 요구나 조건 하나를 빠뜨리고 답하지 않는다.\n" if r["drop_condition"] else "",
        repeat="- 같은 뜻의 문장을 한 번 반복한다.\n" if r["repeat"] else "",
        length=f"- 전체를 {r['max_sentences']}문장 이내로 줄인다.\n" if r["max_sentences"] else "",
        quant=f"- {p.quant_note}\n" if p.quant_note else "",
    )
    return [SystemMessage(system), HumanMessage(f"[이전 대화]\n{_history_text(history)}\n\n[질문]\n{question}\n\n[원래 답변]\n{base}")]


def _history_text(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return "없음"
    return "\n".join(f"{'사용자' if r == 'user' else '챗봇'}: {c[:800]}" for r, c in history)


def strip_rag_marks(text: str) -> str:
    """RAG 답변에서 근거 표기·서비스 안내 문장을 뺀다 (로컬 모델은 근거를 달지 않는다)."""
    text = re.sub(r"\s*\[근거[\s\d,·]+\]", "", text)
    text = re.sub(r"\s*[(\[]계산값[)\]]", "", text)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(("(계산값은", "⚖"))]
    return "\n".join(lines).strip()


def base_answer(question: str, history: list[tuple[str, str]] | None = None) -> str:
    """범용 챗봇의 기준 답 (이전 대화 포함). 체급별 연기의 출발점이다."""
    msgs: list = [SystemMessage(BASE_SYSTEM)]
    for r, c in history or []:
        msgs.append(HumanMessage(c) if r == "user" else AIMessage(c))
    msgs.append(HumanMessage(question))
    return named_chat_model(get_config()["llm"]["persona_model"]).invoke(msgs).content


def answer(model: dict[str, Any], quant: str, question: str, history: list[tuple[str, str]] | None = None,
           base: str | None = None) -> PersonaAnswer:
    """history: 이 모델과의 🎭 대화 기록 (role, content). base를 주면 그 답을 기준 답으로 쓴다 (테스트용)."""
    p = make_persona(model, quant)
    base = strip_rag_marks(base) if base else base_answer(question, history)
    r = p.rule
    if not (r["errors"] or r["drop_condition"] or r["repeat"] or r["max_sentences"]):
        # 큰 체급: 기준 답 그대로. 양자화 말투 지시만으로 고쳐 쓰게 했더니 "능금"을 "능금나무"로 바꾸는 등 내용이 바뀌었다
        return PersonaAnswer(p, base)

    def validate(v: Any) -> tuple[str, list[str]]:
        if not isinstance(v, dict) or not isinstance(v.get("answer"), str) or not v["answer"].strip():
            raise ValueError("answer 없음")
        return v["answer"].strip(), [str(x) for x in v.get("mistakes") or []]

    got = invoke_json(_degrade_messages(p, question, base, history), validate, "흉내답변")
    if got is None:
        return PersonaAnswer(p, base, ["(연출 실패: 기준 답을 그대로 보여 줍니다)"])
    out = PersonaAnswer(p, *got)
    # LLM이 적은 실수 목록은 실제 변경과 다를 수 있어(소형에서 3곳을 바꾸고 2개만 적음), 수치 변경은 코드로 비교한다
    before, after = _numbers(base), _numbers(out.answer)
    out.removed_numbers = [n for n in before if n not in after]
    out.added_numbers = [n for n in after if n not in before]
    return out


def _numbers(text: str) -> list[str]:
    return list(dict.fromkeys(_NUM.findall(_norm_numbers(text))))
