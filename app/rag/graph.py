"""LangGraph RAG 파이프라인: retrieve -> generate (F2007).

외부 문서 수집(collector)은 그래프 앞에서 따로 실행한다. 그래프는 검색과 생성 두 노드만 가진다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.log import get_logger
from app.rag.citations import (NO_CITATION_NOTICE, cited_evidence, cited_numbers, strip_general_numbers,
                               unverified_numbers)
from app.rag.intent import is_license_question
from app.rag.llm import chat_model
from app.rag.prompts import GENERAL_LABEL, LICENSE_DISCLAIMER, answer_messages
from app.rag.retrieve import Evidence, retrieve

log = get_logger("graph")


class RAGState(TypedDict, total=False):
    model_id: str
    question: str
    history: list[tuple[str, str]]  # 같은 모델과의 이전 대화 (role, content). 이어지는 질문 이해용
    calc: str                 # 하드웨어 판정·속도 추정 결과 (계산값). 없으면 빈 문자열
    concept_queries: list[str] | None  # 프롬프트 1의 개념 검색어 ([]면 개념 문서 검색 안 함, None이면 질문으로 검색)
    evidence: list[Evidence]
    answer: str
    cited: list[Evidence]     # 답변에 실제로 인용된 근거
    notice: str               # 인용 근거가 없을 때 안내 문구
    unverified: list[str]     # [근거 n]을 붙였지만 그 근거에 없는 수치가 있는 줄 (환각 경고)


def evidence_block(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"{e.tag} {e.title}\n{e.text}" for e in evidence)


def search_text(state: RAGState) -> str:
    """검색에 쓸 문장. "그럼 Q5는?"처럼 짧은 후속 질문은 직전 사용자 질문을 붙여 주제를 잃지 않게 한다."""
    prev = [c for r, c in (state.get("history") or []) if r == "user"]
    return f"{prev[-1]} {state['question']}" if prev else state["question"]


def retrieve_node(state: RAGState) -> dict[str, Any]:
    return {"evidence": retrieve(state["model_id"], search_text(state), state.get("concept_queries"),
                                 license_question=is_license_question(state["question"]))}


CALC_SOURCE_NOTE = ("(계산값은 [근거 1] 모델 메타데이터의 파라미터 수·레이어 수·KV 헤드 수와 하드웨어 사양표의 "
                    "메모리 대역폭으로 계산한 추정치이며, 실제 런타임과 설정에 따라 다를 수 있습니다.)")


def generate_node(state: RAGState) -> dict[str, Any]:
    messages = answer_messages(state["model_id"], state["question"], evidence_block(state["evidence"]),
                               state.get("calc", ""), state.get("history"))
    answer = chat_model().invoke(messages).content
    answer = answer.replace("[계산값]", "(계산값)")  # 표기 통일: [ ]는 근거 번호 전용
    answer, removed = strip_general_numbers(answer, GENERAL_LABEL)
    if removed:
        log.info("일반 지식 부분에서 근거 없는 수치 문장 제거: %s", removed)
    # 판정·속도(계산값)를 썼는데 그 입력값의 출처인 [근거 1]을 밝히지 않았으면 코드에서 붙인다 (F2010 근거 제시)
    if state.get("calc") and "계산값" in answer and 1 not in cited_numbers(answer):
        answer = f"{answer.rstrip()}\n\n{CALC_SOURCE_NOTE}"
    if is_license_question(state["question"]) and LICENSE_DISCLAIMER not in answer:
        answer = f"{answer.rstrip()}\n\n⚖ {LICENSE_DISCLAIMER}"
    cited = cited_evidence(answer, state["evidence"])
    return {"answer": answer, "cited": cited, "notice": "" if cited else NO_CITATION_NOTICE,
            "unverified": unverified_numbers(answer, state["evidence"], state.get("calc", ""))}


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run(model_id: str, question: str, calc: str = "", concept_queries: list[str] | None = None,
        history: list[tuple[str, str]] | None = None) -> RAGState:
    return build_graph().invoke({"model_id": model_id, "question": question, "calc": calc,
                                 "concept_queries": concept_queries, "history": history or []})


def print_graph() -> str:
    """그래프 구조를 노드/엣지 목록과 Mermaid로 출력한다 (검증 캡처용)."""
    graph = build_graph().get_graph()
    lines = ["노드: " + ", ".join(graph.nodes), "엣지:"]
    lines += [f"  {e.source} -> {e.target}" for e in graph.edges]
    lines += ["", graph.draw_mermaid()]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    print_graph()
