"""모델 추천 상담 (F2011 대화형).

사용자와 몇 번 대화하며 용도·언어·크기·하드웨어 조건을 모은 뒤, 조건 기반 탐색(recommend.search)을 호출해 카드로 보여 준다.
- LLM은 다음 행동만 정한다: 더 물어볼지(ask) / 찾을지(search). 실제 후보 선별과 근거 검증은 recommend.search가 한다.
- 대화에서 말한 GPU·RAM("RTX 3080에 RAM 32기가")은 사양표 이름으로 맞춰 하드웨어로 쓴다.
- 결과 설명 문장은 LLM이 아니라 코드가 카드 내용으로 만든다 (근거 없는 추천 이유가 섞이지 않게).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.data_prep.hw_tables import load_specs, match_gpu
from app.features.hw_detect import Hardware, manual
from app.rag.llm import invoke_json

CONSULT_SYSTEM = """너는 로컬 LLM 추천 상담가다. 사용자가 자기 PC에서 돌릴 오픈 모델을 고르도록 돕는다.
대화로 다음 조건을 파악한다: 용도(코딩, 수학, 대화, 추론, 이미지 이해, 도구 사용), 꼭 지원해야 할 언어, 모델 크기 제한,
그리고 사용자의 PC(GPU, RAM)에서 돌아가야 하는지.

현재 알고 있는 하드웨어: {hw}

규칙
- 용도나 언어 중 하나라도 알면 바로 찾는다(search). 아무 조건도 없을 때만 한두 가지를 짧게 물어본다(ask). 연속으로 두 번 넘게 묻지 않는다.
- 하드웨어를 모르고 사용자가 "내 PC에서"를 원하면, 사이드바의 "하드웨어 감지"를 누르거나 GPU·RAM을 말해 달라고 한다.
- 모델 이름을 직접 추천하거나 사양을 지어내지 않는다. 추천은 검색 결과로만 한다.
- condition에는 사용자가 실제로 말한 조건만 넣는다. 말하지 않은 언어·크기 조건을 추측해 넣지 않는다.
- 출력은 JSON 하나만:
{{"action": "ask" | "search",
  "message": "사용자에게 할 짧은 말 (한국어)",
  "condition": "지금까지 사용자가 말한 조건(용도, 그리고 말했을 때만 언어·크기·내 GPU에서 돌아가야 하는지)을 쉼표로 이은 한 문장, ask일 때도 아는 만큼",
  "gpu": "사용자가 말한 GPU 이름 또는 null", "ram_gb": 숫자 또는 null}}"""


# 언어 조건 이름과 사용자가 그 언어를 말했다고 볼 단서
_LANGUAGES = {"한국어": ("한국", "한글", "korean"), "영어": ("영어", "english"), "일본어": ("일본", "japanese"),
              "중국어": ("중국", "chinese")}


def _drop_unsaid_language(condition: str, history: list[dict[str, str]]) -> str:
    """사용자가 말하지 않은 언어 조건을 condition에서 뺀다.

    프롬프트 예시("한국어 되는 코딩용 모델")를 베껴 "코딩용이면 돼"에도 한국어 조건이 붙어,
    언어 표기가 없는 모델 30건이 빠지고 결과가 0건이 되는 일이 있었다.
    """
    said = " ".join(h["content"] for h in history if h["role"] == "user").lower()
    for lang, cues in _LANGUAGES.items():
        if lang in condition and not any(c in said for c in cues):
            condition = re.sub(rf",?\s*{lang}\s*(?:지원|되는|가능한?|도 되는)?\s*,?", " ", condition)
    return re.sub(r"\s{2,}", " ", condition).strip(" ,")


@dataclass
class ConsultTurn:
    action: str                 # ask | search
    message: str
    condition: str = ""
    hardware: Hardware | None = None   # 대화에서 알아낸 하드웨어 (사양표에 맞춘 것)
    hw_note: str = ""


def _hw_text(hw: Hardware | None) -> str:
    if hw is None:
        return "모름"
    return f"{hw.gpu_name or 'GPU 없음'}, GPU 사용 가능 {hw.gpu_mem_gb:.1f} GB, RAM {hw.ram_gb:.0f} GB"


def step(history: list[dict[str, str]], hw: Hardware | None, asked: int = 0) -> ConsultTurn:
    """history: [{"role": "user"|"assistant", "content": ...}] (마지막이 사용자 발화)
    asked: 마지막 검색 이후 이미 되물은 횟수. 1번 넘게 되묻지 않도록 코드에서 검색으로 바꾼다
    (프롬프트로만 두었더니 "코딩용"이라고 답해도 언어를 또 물었다).
    """
    msgs: list[Any] = [SystemMessage(CONSULT_SYSTEM.format(hw=_hw_text(hw)))]
    for h in history[-10:]:
        msgs.append(HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]))

    def validate(v: Any) -> dict[str, Any]:
        if not isinstance(v, dict) or v.get("action") not in ("ask", "search") or not isinstance(v.get("message"), str):
            raise ValueError("action/message 없음")
        if v["action"] == "search" and not str(v.get("condition") or "").strip():
            raise ValueError("search인데 condition 없음")
        return v

    v = invoke_json(msgs, validate, "추천상담")
    if v is None:
        last = history[-1]["content"]
        return ConsultTurn("search", "말씀하신 조건으로 찾아볼게요.", last)

    turn = ConsultTurn(v["action"], v["message"].strip(), _drop_unsaid_language(str(v.get("condition") or "").strip(), history))
    if turn.action == "ask" and asked >= 1:
        said = " ".join(h["content"] for h in history if h["role"] == "user")
        turn = ConsultTurn("search", "지금까지 말씀하신 조건으로 찾아볼게요.", turn.condition or said)
    gpu = str(v.get("gpu") or "").strip()
    ram = v.get("ram_gb") if isinstance(v.get("ram_gb"), (int, float)) else None
    if gpu or ram:
        spec = match_gpu(gpu) if gpu else None
        if gpu and spec is None:
            turn.hw_note = f"'{gpu}'는 사양표에 없어 하드웨어 판정에 쓰지 못했습니다. 사이드바에서 직접 입력해 주세요."
        else:
            base_ram = ram or (hw.ram_gb if hw else 16)
            turn.hardware = manual(spec.name if spec else (hw.spec_name if hw else None), base_ram)
            turn.hw_note = f"대화에서 알려 주신 하드웨어({_hw_text(turn.hardware)})로 판정합니다."
            if ram is None and hw is None:
                turn.hw_note += " RAM은 16GB로 가정했습니다."
    # GPU를 알려 줬다는 건 그 GPU에서 돌아가야 한다는 뜻이다 (LLM이 condition에서 빠뜨리는 경우가 있어 코드로 붙인다)
    if turn.hardware is not None and turn.condition and "GPU" not in turn.condition:
        turn.condition += ", 내 GPU에서 돌아가는 거"
    return turn


def gpu_names() -> list[str]:
    return [s.name for s in load_specs()]
