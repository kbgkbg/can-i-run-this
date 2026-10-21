"""질문 의도 판별 (규칙 기반). 수집·생성 단계가 같은 판별을 쓰도록 한곳에 둔다."""
from __future__ import annotations

import re

LICENSE_QUESTION = re.compile(r"라이선스|라이센스|license|상업|상용|commercial|이용\s*조건|사용\s*조건|재배포|약관", re.IGNORECASE)


def is_license_question(question: str) -> bool:
    """라이선스·이용 조건 질문이면 라이선스 원문을 수집하고, 답변에 면책 안내를 붙인다 (F2012)."""
    return bool(LICENSE_QUESTION.search(question))
