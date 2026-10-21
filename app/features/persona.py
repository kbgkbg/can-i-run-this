"""모델 흉내 답변 (체험용 연기).

고른 오픈 모델을 실제로 돌리지 않고(호스팅 추론 비용 없이), 같은 LLM(GMS)으로 그 모델 체급의 답을 흉내 낸다.
1단계: 기준 답. 채팅에서는 방금 만든 RAG 답변(근거 기반, 근거 표기 제거)을 쓴다. 기준 답이 없으면 근거 없이 답한다.
       근거 없이 쓴 기준 답은 처음부터 틀려서(예: "8B 4비트 → 1GB") 큰 체급까지 엉터리가 되고 체급 차이가 안 보였다.
2단계: 체급에 맞게 고쳐 쓴다. "작은 모델처럼 틀려라"는 지시는 잘 지켜지지 않아(0.6B 역할도 계산을 다 맞힘),
       넣을 사실·계산 오류 개수, 조건 누락, 반복, 문장 수를 정확히 지정하고 넣은 실수 목록을 함께 받는다.
체급(유효 파라미터 구간)과 실수 개수는 가정값이다 (config.yaml persona). 결과는 실제 모델 출력이 아니다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_config
from app.features.speed import active_params
from app.rag.citations import _NUM, _norm_numbers
from app.rag.llm import chat_model, invoke_json

PERSONA_NOTICE = "🎭 연기(시뮬레이션)입니다. 실제 이 모델의 출력이 아니며, 내용의 사실 여부를 믿지 마세요."

BASE_SYSTEM = "너는 인터넷 검색 없이 자기 지식으로만 답하는 로컬 LLM이다. 한국어로 간결하게 답한다."

DEGRADE_SYSTEM = """너는 LLM 체험 서비스의 연출 담당이다. 주어진 답변을 "{tier} 체급(유효 파라미터 약 {params_b}B) 로컬 모델이 썼을 법한 답"으로 고쳐 쓴다.

정확히 지킬 것
- 사실·수치·계산 중 정확히 {errors}곳을 그럴듯하게 틀린 값으로 바꾼다. 틀린 곳을 표시하거나 암시하지 않는다.
{drop}{repeat}{length}{quant}
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


def _degrade_messages(p: Persona, question: str, base: str) -> list:
    r = p.rule
    system = DEGRADE_SYSTEM.format(
        tier=p.tier, params_b=p.params_b, errors=r["errors"],
        drop="- 질문 속 요구나 조건 하나를 빠뜨리고 답하지 않는다.\n" if r["drop_condition"] else "",
        repeat="- 같은 뜻의 문장을 한 번 반복한다.\n" if r["repeat"] else "",
        length=f"- 전체를 {r['max_sentences']}문장 이내로 줄인다.\n" if r["max_sentences"] else "",
        quant=f"- {p.quant_note}\n" if p.quant_note else "",
    )
    return [SystemMessage(system), HumanMessage(f"[질문]\n{question}\n\n[원래 답변]\n{base}")]


def strip_rag_marks(text: str) -> str:
    """RAG 답변에서 근거 표기·서비스 안내 문장을 뺀다 (로컬 모델은 근거를 달지 않는다)."""
    text = re.sub(r"\s*\[근거[\s\d,·]+\]", "", text)
    text = re.sub(r"\s*[(\[]계산값[)\]]", "", text)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(("(계산값은", "⚖"))]
    return "\n".join(lines).strip()


def answer(model: dict[str, Any], quant: str, question: str, base: str | None = None) -> PersonaAnswer:
    p = make_persona(model, quant)
    base = strip_rag_marks(base) if base else chat_model().invoke([SystemMessage(BASE_SYSTEM), HumanMessage(question)]).content
    r = p.rule
    if not (r["errors"] or r["drop_condition"] or r["repeat"] or r["max_sentences"] or p.quant_note):
        return PersonaAnswer(p, base)  # 큰 체급: 근거 없는 답 그대로 (그 자체로 근거 없는 모델의 답)

    def validate(v: Any) -> tuple[str, list[str]]:
        if not isinstance(v, dict) or not isinstance(v.get("answer"), str) or not v["answer"].strip():
            raise ValueError("answer 없음")
        return v["answer"].strip(), [str(x) for x in v.get("mistakes") or []]

    got = invoke_json(_degrade_messages(p, question, base), validate, "흉내답변")
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
