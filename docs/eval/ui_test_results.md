# 화면 기능 점검 결과 (Streamlit AppTest)

- 실행: `python scripts/ui_checks.py` · 통과 13/13

| 요구사항 | 점검 내용 | 결과 | 실행 결과 |
|---|---|---|---|
| F2009 | 모델별 대화 분리: 8B에서 질문 → 4B 전환 → 8B 복귀 | PASS | 8B 메시지 2 → 4B ['assistant'] (안내 말풍선만) → 8B 2 |
| F2008 | 답변 아래에 실제 인용된 근거만 원문 링크와 표시 | PASS | **[근거 4]** Qwen/Qwen3-8B-GGUF 모델 카드 · HF 모델 카드 · [원문](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| F2012 | 진행 상태 표시 후 완료로 바뀜 | PASS | status=['답변 완료'] |
| F2012 | API 키 누락 안내 | PASS | 실행에 필요한 설정이 없습니다: OPENAI_API_KEY `.env.example`을 `.env`로 복사한 뒤 값을 채우고 앱을 다시 실행하 |
| F2012 | 색인 0건이면 색인 스크립트 실행 안내 | PASS | 모델 메타데이터 색인이 비어 있습니다 (0건). 터미널에서 `python scripts/build_meta_index.py`를 먼저 실행하세요 |
| NF2004 | 잘못된 키: 사용자 문구만 표시, 화면 예외 없음 | PASS | ⚠ API 키가 올바르지 않습니다. .env의 OPENAI_API_KEY를 확인하세요. |
| F2010 | 컨텍스트 32k 설정 후 최대 2k 모델로 전환해도 예외 없음, 돌아오면 값 복원 | PASS | 전환 후 2048, 복귀 후 32768, 예외 0 |
| F2010 | 🎭 대화 체험: 문서 검색 없이 아무 주제나 답하고 이전 대화를 기억, 근거 Q&A 기록과 분리 | PASS | 안내 ['🎭'], 문서 수집 상태 0개, 말풍선 ['user', '🎭', 'user', '🎭'], 답1 ['**능금**은 옛날에 사과를 이르던 말이기도 하고, 능금나무에서 열리는 작은 열매를 뜻하기도 해요. 보통 오늘날의 사과보다 작고 신맛이 강한 재래 과일을 가리킵니다. \n그래서 문'], 후속 답 ['‘능금’이 무슨 뜻인지 물어봤어요.'], 끄면 ['assistant'] |
| F2011 | 추천 상담: 모델 이름으로 물으면 그 모델의 내 PC 판정 카드, 안 맞으면 같은 계열 대안 (사양은 그대로 유지) | PASS | ['**Qwen/Qwen3.8-Flash-Next**은(는) Apple M5 Max (40-core GPU)에서 쓰기 어려워요: 불가 (어떤 양자화로도 메모리에 안 올라감) (계산값).\n\n대신 같은 계열에서 이 / 카드 ['**Qwen/Qwen3.8-Flash-Next**', '**Qwen/Qwen3.8-27B**'] / 사양 Apple M5 Max (40-core GPU) |
| F2010 | 속도 체험: 사양을 건너뛰고 모델만 골라도 켜면 자동 감지 후 재생, 예전 TODO였던 RTX 4070은 사양표 대역폭으로 재생 | PASS | 자동 감지(none) 후 ['⏱ 내 PC 추정 속도 32.8 tok/s로 재생 중 (추정치, 실제와 다를 수 있음)'] / 직접 입력 ['RTX 4070', 504.0, 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'] 후 ['⏱ 내 PC 추정 속도 153.3 tok/s로 재생 중 (추정치, 실제와 다를 수 있음)'] |
| F2009 | 빈 대화: 안내 말풍선과 첫 질문 소요 안내, 예시 질문 버튼으로 바로 질문 | PASS | OpenGVLab/InternVL2-1B: 질문 전 ['assistant'], 첫 질문 안내=True, 진행 상태 ['답변 완료 · 첫 질문이라 근거 문서 수집 포함 17초'] → 질문 ['권장 샘플링 설정은?'] 후 ['user', 'assistant'] |
| F2011 | 추천 상담: 되묻기 1회 후 검색, 대화 속 하드웨어 적용, 카드 클릭 시 대화로 전환 | PASS | 카드 1개, 하드웨어 RTX 3080, 전환 후 모델에게 묻기 / Qwen/Qwen2.5-Coder-7B-Instruct |
| F2010 | 첫 화면 사양 선택(Mac 탭): M3 Max 40코어·36GB → 통합 메모리 판정, 컨텍스트별 차이 표 | PASS | 첫 화면 True → 적용 후 False, 하드웨어 ['apple', 'Apple M3 Max (40-core GPU)', 25.2, 400.0], 판정 ['🟢 **Q5_K_M: GPU에 전부 올라감** — 필요 22.9 GB, 추정 약 94 tok/s (계산값, 컨텍스트 표 6행 |
