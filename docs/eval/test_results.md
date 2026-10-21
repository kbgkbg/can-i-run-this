# 자동 점검 결과 (테스트 결과표)

- 실행: `python scripts/run_checks.py` · 2026-10-21 08:52 · 통과 22/22
- 기준 모델 `Qwen/Qwen3-8B`, LLM `gpt-4o-mini`, 임베딩 `text-embedding-3-small`

| 요구사항 | 점검 내용 | 결과 | 근거(실행 결과) |
|---|---|---|---|
| F2001 | 항목 메타데이터 100건 이상 로드, 항목 1건 = 텍스트 1개 | PASS | 150건, 필드 7개 포함 (0.0s) |
| F2002 | 색인 청크 수 = 항목 수, 재색인해도 불변 | PASS | 색인 전 150 -> 후 150, 항목 150 (5.3s) |
| F2003 | 프롬프트 1이 파싱 가능한 검색어 배열을 낸다 (모델 ID 포함) | PASS | ['Qwen/Qwen3-8B', 'Qwen3-8B GGUF'] (2.6s) |
| F2004·NF2002 | 이미 수집한 검색어는 다시 수집하지 않는다 (캐시 히트) | PASS | cache_hit=True, 문서 7건 사용 (3.0s) |
| F2004 | 선택 모델의 모델 카드는 항상 색인되어 있다 | PASS | Qwen/Qwen3-8B 색인 문서 7건 (0.0s) |
| F2005 | model_docs 청크 ID = 모델ID:문서ID:순번, 메타데이터 필드 저장 | PASS | 청크 50개 검사, 예: Qwen/Qwen3-8B:hf_card:Qwen/Qwen3-8B:0 (0.1s) |
| F2005 | 청크 크기·중복은 config.yaml에서 읽는다 | PASS | chunk_size=1000, overlap=200 (0.0s) |
| F2006 | 메타데이터 근거가 항상 [근거 1], Top-k를 바꾸면 근거 수가 바뀐다 | PASS | top_k=2 -> 근거 3건, top_k=6 -> 7건 (2.0s) |
| F2006 | 청크 ID·본문 해시·근사 중복 제거 | PASS | 근거 8건, 중복 ID 없음 (0.9s) |
| F2007 | LangGraph가 retrieve -> generate 두 노드 | PASS | 노드 ['retrieve', 'generate'], 엣지 [('__start__', 'retrieve'), ('retrieve', 'generate'), ('generate', '__end__')] (0.0s) |
| F2007 | 문서에 없는 질문에 추측하지 않고 '확인되지 않습니다' | PASS | 제공된 문서에서 확인되지 않습니다. Qwen3-8B 모델의 학습에 사용된 GPU 시간이나 비용에 대한 정보는 포함되어 있지 않습니다. (1.6s) |
| F2008 | 실제 인용된 근거만 표시, 인용이 없으면 안내 | PASS | [근거 3][근거 9] -> [3], 없음 안내 '이 답변에는 인용된 근거가 없습니다.' (1.9s) |
| F2008 | 근거에 없는 수치를 인용 형식으로 쓰면 경고 (환각 검출) | PASS | ['- TopK: 50 [근거 1] (근거에 없는 수치: 50)'] (0.0s) |
| F2010 | 판정·속도 계산 (Qwen3-8B, RTX 3080, 컨텍스트 4096) | PASS | Q4_K_M 5.8 GB GPU에 전부 올라감, 89.6 tok/s (0.0s) |
| F2011 | 조건 탐색 결과가 조건(용도 code)에 맞는 모델만 | PASS | Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-14B-Instruct (4.8s) |
| F2012 | 라이선스 질문에 면책 안내 | PASS | 라이선스 조건을 준수해야 함 [근거 3] 법률 자문이 아니며, 정확한 조건은 원문 라이선스를 확인하세요. (2.6s) |
| NF2001 | 대표 질문 응답 시간 (수집 캐시 히트 상태) | PASS | 4.6초 (단계별 수집 1.2s, 검색 0.5s, 생성 2.9s) (4.6s) |
| NF2003 | API 키가 커밋된 파일에 없다 | PASS | 추적 파일 79개 중 키 포함 0개, .env gitignore=True (0.0s) |
| NF2004 | 오류는 사용자 문구로 바꾸고 스택 트레이스를 노출하지 않는다 | PASS | 답변을 만드는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. (0.0s) |
| NF2005 | 수집·색인·검색·생성·UI 모듈 분리, 도메인 의존 코드는 data_prep | PASS | app/data_prep, app/collect, app/index, app/rag, app/features, app/ui (0.0s) |
| NF2006 | User-Agent 연락처, 호스트별 호출 간격(arXiv 3초), 재시도 | PASS | can-i-run-this/0.1.0 (https://github.com/kbgkbg/can-i-run-this) python-requests/2.34.2 (0.0s) |
| NF2007 | 데이터·문서 출처와 이용 조건 기록 | PASS | data/SOURCES.md (0.0s) |
