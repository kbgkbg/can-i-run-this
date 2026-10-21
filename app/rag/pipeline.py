"""질문 1건 처리: 외부 문서 수집 -> LangGraph(retrieve -> generate).

UI(Streamlit)와 나중의 다른 화면이 같은 진입점을 쓰도록 화면 코드와 분리했다 (NF2005).
예외는 사용자에게 보여도 되는 안내 문구로 바꾸고, 상세는 로그 파일에만 남긴다 (NF2004).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import openai

from app.collect.collector import CollectResult, collect
from app.log import get_logger
from app.rag.graph import build_graph
from app.rag.retrieve import Evidence

log = get_logger("pipeline")

StepCallback = Callable[[str, str], None]


@dataclass
class AskResult:
    answer: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    cited: list[Evidence] = field(default_factory=list)
    notice: str = ""                                   # 인용 근거 없음 안내
    collect: CollectResult | None = None
    warnings: list[str] = field(default_factory=list)  # 답은 냈지만 알려야 할 것 (수집 일부 실패, 문서 없음 등)
    error: str = ""                                    # 답을 내지 못한 이유 (사용자용 문구)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error


def user_message(e: Exception) -> str:
    """예외 -> 사용자 안내 문구. 원문 메시지·스택 트레이스는 화면에 내보내지 않는다."""
    if isinstance(e, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. .env의 OPENAI_API_KEY를 확인하세요."
    if isinstance(e, openai.RateLimitError):
        return "LLM 호출 한도에 걸렸습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "LLM 서버(GMS)에 연결하지 못했습니다. 네트워크와 .env의 OPENAI_BASE_URL을 확인하세요."
    if isinstance(e, openai.APIStatusError):
        return f"LLM 서버가 오류를 돌려주었습니다 (HTTP {e.status_code}). 잠시 후 다시 시도해 주세요."
    return "답변을 만드는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


def _noop(stage: str, message: str) -> None:
    pass


def ask(model: dict[str, Any], question: str, calc: str = "", on_step: StepCallback = _noop) -> AskResult:
    result = AskResult()
    t0 = time.monotonic()

    # 1) 외부 문서 수집. 실패해도 이미 색인된 문서로 답을 이어간다.
    concept_queries: list[str] | None = None
    try:
        result.collect = collect(model, question, on_step=on_step)
        concept_queries = result.collect.concept_queries
        result.warnings += result.collect.errors
    except Exception as e:  # noqa: BLE001 - 어떤 실패든 사용자 문구로 바꾸고 계속한다
        log.exception("문서 수집 실패: %s", model["id"])
        result.warnings.append("외부 문서 수집에 실패해 이미 수집된 문서만으로 답합니다. (" + user_message(e) + ")")
    result.timings["수집"] = time.monotonic() - t0

    # 2) retrieve -> generate. 노드가 끝날 때마다 진행 상태를 알린다.
    t1 = time.monotonic()
    state: dict[str, Any] = {}
    try:
        on_step("검색", "질문과 관련된 근거를 찾는 중")
        inputs = {"model_id": model["id"], "question": question, "calc": calc, "concept_queries": concept_queries}
        for update in build_graph().stream(inputs, stream_mode="updates"):
            for node, values in update.items():
                state.update(values)
                if node == "retrieve":
                    result.timings["검색"] = time.monotonic() - t1
                    on_step("생성", f"근거 {len(values['evidence'])}건으로 답변 생성 중")
    except Exception as e:  # noqa: BLE001
        log.exception("검색/생성 실패: %s", model["id"])
        result.error = user_message(e)
        return result
    result.timings["생성"] = time.monotonic() - t1 - result.timings.get("검색", 0.0)

    result.answer = state.get("answer", "")
    result.evidence = state.get("evidence", [])
    result.cited = state.get("cited", [])
    result.notice = state.get("notice", "")
    if not any(e.source_type != "model_meta" for e in result.evidence):
        result.warnings.append("관련 외부 문서를 찾지 못해 모델 메타데이터만 근거로 답했습니다.")
    log.info("질문 처리 %s | %s", model["id"], {k: round(v, 1) for k, v in result.timings.items()})
    return result
