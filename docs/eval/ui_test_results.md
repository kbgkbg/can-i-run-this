# 화면 기능 점검 결과 (Streamlit AppTest)

- 실행: `python scripts/ui_checks.py` · 통과 7/7

| 요구사항 | 점검 내용 | 결과 | 실행 결과 |
|---|---|---|---|
| F2009 | 모델별 대화 분리: 8B에서 질문 → 4B 전환 → 8B 복귀 | PASS | 8B 메시지 2 → 4B 0 → 8B 2 |
| F2008 | 답변 아래에 실제 인용된 근거만 원문 링크와 표시 | PASS | **[근거 4]** Qwen/Qwen3-8B-GGUF 모델 카드 · HF 모델 카드 · [원문](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| F2012 | 진행 상태 표시 후 완료로 바뀜 | PASS | status=['답변 완료'] |
| F2012 | API 키 누락 안내 | PASS | 실행에 필요한 설정이 없습니다: OPENAI_API_KEY `.env.example`을 `.env`로 복사한 뒤 값을 채우고 앱을 다시 실행하 |
| F2012 | 색인 0건이면 색인 스크립트 실행 안내 | PASS | 모델 메타데이터 색인이 비어 있습니다 (0건). 터미널에서 `python scripts/build_meta_index.py`를 먼저 실행하세요 |
| NF2004 | 잘못된 키: 사용자 문구만 표시, 화면 예외 없음 | PASS | ⚠ API 키가 올바르지 않습니다. .env의 OPENAI_API_KEY를 확인하세요. |
| F2010 | 컨텍스트 32k 설정 후 4k 모델로 전환해도 예외 없음, 돌아오면 값 복원 | PASS | 전환 후 4096, 복귀 후 32768, 예외 0 |
