"""LangGraph RAG 파이프라인: retrieve -> generate (F2007).

외부 문서 수집(collector)은 그래프 앞에서 따로 실행한다. 그래프는 검색과 생성 두 노드만 가진다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag.citations import NO_CITATION_NOTICE, cited_evidence
from app.rag.llm import chat_model
from app.rag.prompts import answer_messages
from app.rag.retrieve import Evidence, retrieve


class RAGState(TypedDict, total=False):
    model_id: str
    question: str
    calc: str                 # 하드웨어 판정·속도 추정 결과 (계산값). 없으면 빈 문자열
    evidence: list[Evidence]
    answer: str
    cited: list[Evidence]     # 답변에 실제로 인용된 근거
    notice: str               # 인용 근거가 없을 때 안내 문구


def evidence_block(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"{e.tag} {e.title}\n{e.text}" for e in evidence)


def retrieve_node(state: RAGState) -> dict[str, Any]:
    return {"evidence": retrieve(state["model_id"], state["question"])}


def generate_node(state: RAGState) -> dict[str, Any]:
    messages = answer_messages(state["model_id"], state["question"], evidence_block(state["evidence"]), state.get("calc", ""))
    answer = chat_model().invoke(messages).content
    cited = cited_evidence(answer, state["evidence"])
    return {"answer": answer, "cited": cited, "notice": "" if cited else NO_CITATION_NOTICE}


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run(model_id: str, question: str, calc: str = "") -> RAGState:
    return build_graph().invoke({"model_id": model_id, "question": question, "calc": calc})


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
