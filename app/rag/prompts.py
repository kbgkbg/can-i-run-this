"""프롬프트 3개 (F2003, F2004, F2007).

1. 검색어 추출: 모델 메타데이터 + 질문 -> JSON 배열
2. 후보 선별: 후보의 제목·요약만 보고 색인할 문서를 고른다 -> {"selected": [...], "reason": "..."}
3. 답변 생성: 주어진 근거만 사용, [근거 n] 표기 (Phase 4)
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------- 프롬프트 1
QUERY_EXTRACTION_SYSTEM = """너는 로컬 LLM 배포 전문가다. 사용자가 특정 오픈 모델을 자기 PC에서 돌리려 할 때,
질문에 답하는 데 필요한 외부 문서를 찾기 위한 검색어를 만든다.

검색어는 형태에 따라 출처가 자동으로 정해진다.
- 저장소 검색어 -> Hugging Face 모델 저장소 이름 검색: 모델 ID("Qwen/Qwen3-8B") 또는 "모델명 GGUF"처럼
  모델명이나 양자화 형식(GGUF, AWQ, GPTQ, MLX)이 들어간 검색어. 양자화·메모리 질문에는 "모델명 GGUF"가 유용하다.
- 개념 검색어 -> 위키백과 검색: 그 외 모든 검색어. "컨텍스트 윈도", "트랜스포머 (기계 학습)"처럼 분야가 드러나게 쓴다.
  "양자화"처럼 다른 분야(물리학)와 겹치는 말은 피한다.

규칙
- 검색어는 {max_queries}개 이하, 각각 짧은 키워드 (문장 금지). 질문에 필요한 것만 뽑는다.
- 모델 자체에 대한 질문이면 모델 ID를 넣는다.
- GPU 같은 하드웨어 이름은 검색어로 쓰지 않는다 (하드웨어 사양은 서비스가 별도 표로 계산한다).
- 출력은 JSON 문자열 배열 하나만. 설명, 코드 블록 표시 금지.
예: ["Qwen/Qwen3-32B", "Qwen3-32B GGUF", "컨텍스트 윈도"]"""

QUERY_EXTRACTION_USER = """[선택한 모델 메타데이터]
{model_text}

[사용자 질문]
{question}"""


def query_extraction_messages(model_text: str, question: str, max_queries: int) -> list[BaseMessage]:
    return [
        SystemMessage(QUERY_EXTRACTION_SYSTEM.format(max_queries=max_queries)),
        HumanMessage(QUERY_EXTRACTION_USER.format(model_text=model_text, question=question)),
    ]


# ---------------------------------------------------------------- 프롬프트 2
CANDIDATE_SELECTION_SYSTEM = """너는 로컬 LLM 배포 전문가이자 자료 조사 담당이다.
검색으로 모은 후보 문서의 제목과 짧은 요약만 보고, 사용자 질문에 답하는 데 근거가 될 문서를 고른다.
본문을 받기 전에 관련 없는 문서를 걸러 외부 호출을 줄이는 것이 목적이다.
선택한 모델의 공식 모델 카드는 이미 따로 포함되어 있다.

고를 것
- 선택한 모델과 같은 모델의 양자화 저장소(예: 같은 이름의 GGUF) 카드: 양자화·파일 크기·메모리 질문일 때.
  같은 제작사(모델 ID의 "/" 앞부분)가 올린 공식 저장소를 개인이 올린 저장소보다 우선한다.
- 질문의 개념과 같은 분야(기계 학습·LLM)의 위키백과 문서.

버릴 것
- 다른 모델의 저장소: 이름이 비슷해도 세대·크기·종류(VL, Coder 등)가 다르면 다른 모델이다.
- 공식 모델 카드 내용을 거의 그대로 반복하는 변형(-Base, -AWQ, -FP8 등): 질문이 그 형식에 대한 것일 때만 고른다.
- 이름만 같은 다른 분야 문서 (예: 물리학의 양자화).

규칙
- 최대 {max_selected}개. 확신이 없으면 고르지 않는다. 관련 문서가 없으면 빈 배열을 낸다.
- 출력은 JSON 객체 하나만: {{"selected": ["후보 id", ...], "reason": "고른 이유와 버린 이유를 한두 문장으로"}}"""

CANDIDATE_SELECTION_USER = """[선택한 모델] {model_id}

[사용자 질문]
{question}

[후보 목록] (id / 출처 / 제목 / 요약)
{candidates}"""


def candidate_selection_messages(
    model_id: str, question: str, candidates: list[dict[str, Any]], max_selected: int
) -> list[BaseMessage]:
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in candidates)
    return [
        SystemMessage(CANDIDATE_SELECTION_SYSTEM.format(max_selected=max_selected)),
        HumanMessage(CANDIDATE_SELECTION_USER.format(model_id=model_id, question=question, candidates=lines)),
    ]
