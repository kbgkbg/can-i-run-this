"""외부 API 공통 호출 (NF2006).

- User-Agent에 프로젝트명과 연락처를 넣는다 (위키백과 User-Agent 정책).
- 호스트별 최소 호출 간격을 지킨다 (arXiv 3초 등). 재시도도 같은 간격 규칙을 거친다.
- 429/5xx와 연결 오류는 지수 백오프로 재시도하고, Retry-After 헤더가 있으면 따른다.
- 실패하면 ExternalAPIError를 던진다. 상세는 로그에만 남기고 화면에는 안내 문구만 쓴다 (NF2004).
"""
from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

from app.config import env, get_config
from app.log import get_logger

log = get_logger("http")


class ExternalAPIError(Exception):
    """외부 API 호출 실패. str()은 사용자에게 보여도 되는 문구만 담는다."""

    def __init__(self, host: str, reason: str):
        super().__init__(f"외부 서비스({host})에 연결하지 못했습니다: {reason}")
        self.host = host


_session: requests.Session | None = None
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def user_agent() -> str:
    cfg = get_config()["project"]
    contact = env("CONTACT_INFO") or "contact-not-set"
    return f"{cfg['name']}/{cfg['version']} (SSAFY AI core project; {contact}) python-requests"


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = user_agent()
    return _session


def _wait_for_slot(host: str) -> None:
    """같은 호스트에 대한 직전 호출 이후 min_interval_s가 지날 때까지 기다린다."""
    intervals = get_config()["http"]["min_interval_s"]
    interval = float(intervals.get(host, intervals["default"]))
    with _lock:
        now = time.monotonic()
        wait = _last_call.get(host, 0.0) + interval - now
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def _retry_after_seconds(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if value and value.isdigit():
        return float(value)
    return None


def request(method: str, url: str, *, headers: dict[str, str] | None = None, **kwargs: Any) -> requests.Response:
    """재시도·호출 간격·User-Agent를 적용한 HTTP 요청. 2xx가 아니면 ExternalAPIError."""
    cfg = get_config()["http"]
    host = urlparse(url).netloc
    kwargs.setdefault("timeout", cfg["timeout_s"])
    retry_statuses = set(cfg["retry_statuses"])
    attempts = cfg["max_retries"] + 1

    last_reason = "알 수 없는 오류"
    for attempt in range(attempts):
        _wait_for_slot(host)
        try:
            resp = _get_session().request(method, url, headers=headers, **kwargs)
        except requests.RequestException as e:
            last_reason = "네트워크 오류"
            log.warning("%s %s 실패 (시도 %d/%d): %r", method, url, attempt + 1, attempts, e)
            delay = None
        else:
            if resp.ok:
                return resp
            last_reason = f"HTTP {resp.status_code}"
            log.warning("%s %s -> %d (시도 %d/%d)", method, url, resp.status_code, attempt + 1, attempts)
            if resp.status_code not in retry_statuses:
                break  # 404, 401 같은 오류는 재시도해도 같으므로 바로 실패
            delay = _retry_after_seconds(resp)

        if attempt < attempts - 1:
            time.sleep(delay if delay is not None else cfg["backoff_factor"] * (2 ** attempt))

    raise ExternalAPIError(host, last_reason)


def get(url: str, **kwargs: Any) -> requests.Response:
    return request("GET", url, **kwargs)


def get_json(url: str, **kwargs: Any) -> Any:
    return get(url, **kwargs).json()
