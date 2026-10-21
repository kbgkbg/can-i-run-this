"""질문 의도 판별 (규칙 기반). 수집·생성 단계가 같은 판별을 쓰도록 한곳에 둔다."""
from __future__ import annotations

import re

LICENSE_QUESTION = re.compile(r"라이선스|라이센스|license|상업|상용|commercial|이용\s*조건|사용\s*조건|재배포|약관", re.IGNORECASE)


def is_license_question(question: str) -> bool:
    """라이선스·이용 조건 질문이면 라이선스 원문을 수집하고, 답변에 면책 안내를 붙인다 (F2012)."""
    return bool(LICENSE_QUESTION.search(question))


PAPER_QUESTION = re.compile(
    r"학습|훈련|사전\s*학습|데이터셋|학습\s*데이터|토큰\s*수|벤치마크|성능\s*비교|점수|평가|아키텍처|구조|논문|기술\s*보고서|"
    r"train|dataset|benchmark|architecture|paper|technical\s*report|MMLU|GSM8K|HumanEval", re.IGNORECASE)


def is_paper_question(question: str) -> bool:
    """학습·벤치마크·구조처럼 모델 카드보다 기술 보고서(arXiv)에 자세한 내용을 묻는 질문인지 (F2016)."""
    return bool(PAPER_QUESTION.search(question))
