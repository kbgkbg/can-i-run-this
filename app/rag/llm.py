"""GMS(OpenAI 호환) 채팅 모델과 JSON 응답 처리."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Callable

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.config import env, get_config
from app.log import get_logger

log = get_logger("llm")


@lru_cache(maxsize=1)
def chat_model() -> ChatOpenAI:
    cfg = get_config()["llm"]
    return ChatOpenAI(
        model=cfg["chat_model"],
        temperature=cfg["temperature"],
        openai_api_key=env("OPENAI_API_KEY"),
        openai_api_base=env("OPENAI_BASE_URL"),
        request_timeout=cfg["request_timeout_s"],
        max_retries=cfg["max_retries"],
    )


def _extract_json(text: str) -> Any:
    """```json 코드 블록이나 앞뒤 설명이 붙어 와도 첫 JSON 값만 꺼낸다."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("JSON 없음")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def invoke_json(messages: list[BaseMessage], validate: Callable[[Any], Any], label: str) -> Any | None:
    """LLM을 호출해 JSON으로 파싱하고 validate()를 통과한 값을 반환한다.

    파싱/검증에 실패하면 형식을 다시 요구하며 1회 재요청하고, 그래도 실패하면 None (호출 측이 기본값으로 대체).
    """
    attempt_messages = list(messages)
    for attempt in range(2):
        reply = chat_model().invoke(attempt_messages)
        try:
            return validate(_extract_json(reply.content))
        except (ValueError, TypeError, KeyError) as e:
            log.warning("%s JSON 파싱 실패 (시도 %d): %s | 응답 앞부분: %r", label, attempt + 1, e, reply.content[:200])
            attempt_messages = list(messages) + [
                reply,
                HumanMessage("직전 응답은 요구한 JSON 형식이 아닙니다. 설명 없이 요구한 JSON만 다시 출력하세요."),
            ]
    return None
