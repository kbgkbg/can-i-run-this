"""설정 로드.

- 조정값: app/config.yaml
- 비밀 값: .env (NF2003) -> 환경변수로만 읽고, 로그/화면에 원문을 내보내지 않는다.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# 화면/로그에 노출하면 안 되는 환경변수
SECRET_KEYS = ("OPENAI_API_KEY", "HF_TOKEN")


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """config.yaml을 읽어 dict로 반환한다 (프로세스당 1회)."""
    load_dotenv(PROJECT_ROOT / ".env")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def reload_config() -> dict[str, Any]:
    """검증 중 config.yaml을 바꿔 다시 읽을 때 사용한다."""
    get_config.cache_clear()
    return get_config()


def path(key: str) -> Path:
    """paths.<key>를 프로젝트 루트 기준 절대 경로로 반환한다."""
    return PROJECT_ROOT / get_config()["paths"][key]


def env(name: str) -> str | None:
    """환경변수 값을 반환한다. 비어 있으면 None."""
    get_config()  # .env 로드 보장
    value = os.getenv(name, "").strip()
    return value or None


def missing_required_env() -> list[str]:
    """실행에 필요한데 비어 있는 환경변수 이름 목록 (F2012 키 누락 안내용)."""
    required = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CONTACT_INFO")
    return [name for name in required if env(name) is None]


def describe() -> dict[str, Any]:
    """비밀 값을 가린 설정 요약 (디버깅/검증 출력용)."""
    return {
        "config": get_config(),
        "env": {
            name: ("설정됨" if env(name) else "없음")
            for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HF_TOKEN", "CONTACT_INFO")
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), ensure_ascii=False, indent=2))
