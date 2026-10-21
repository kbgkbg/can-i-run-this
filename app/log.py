"""로그 설정 (NF2004).

상세 오류(스택 트레이스)는 로그 파일(logs/app.log)에만 남기고, 화면에는 안내 문구만 띄운다.
"""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler

from app.config import SECRET_KEYS, env, path

_configured = False


class _RedactSecrets(logging.Filter):
    """로그 메시지에 비밀 값이 섞여 들어가면 가린다."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for name in SECRET_KEYS:
            value = env(name)
            if value and value in msg:
                msg = msg.replace(value, f"<{name}>")
        # Authorization 헤더 형태도 가린다
        msg = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", msg)
        record.msg, record.args = msg, ()
        return True


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        log_file = path("log_file")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(_RedactSecrets())
        root = logging.getLogger("app")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _configured = True
    return logging.getLogger(f"app.{name}" if not name.startswith("app") else name)
