# 자동 점검 결과 (테스트 결과표)

- 실행: `python scripts/run_checks.py` · 2026-10-22 08:30 · 통과 37/37
- 기준 모델 `Qwen/Qwen3-8B`, LLM `gpt-6-sol`, 임베딩 `text-embedding-3-large`

| 요구사항 | 점검 내용 | 결과 | 근거(실행 결과) |
|---|---|---|---|
| F2001 | 항목 메타데이터 100건 이상 로드, 항목 1건 = 텍스트 1개 | PASS | 150건, 필드 7개 포함 (0.0s) |
| F2002 | 색인 청크 수 = 항목 수, 재색인해도 불변 | PASS | 색인 전 150 -> 후 150, 항목 150 (7.0s) |
| F2003 | 프롬프트 1이 파싱 가능한 검색어 배열을 낸다 (모델 ID 포함) | PASS | ['Qwen/Qwen3-8B', 'Qwen3-8B GGUF'] (2.5s) |
| F2004·NF2002 | 이미 수집한 검색어는 다시 수집하지 않는다 (캐시 히트) | PASS | cache_hit=True, 문서 7건 사용 (4.5s) |
| F2004 | 선택 모델의 모델 카드는 항상 색인되어 있다 | PASS | Qwen/Qwen3-8B 색인 문서 7건 (0.0s) |
| F2005 | model_docs 청크 ID = 모델ID:문서ID:순번, 메타데이터 필드 저장 | PASS | 청크 50개 검사, 예: Qwen/Qwen3-8B:hf_card:Qwen/Qwen3-8B:0 (4.6s) |
| F2005 | 청크 크기·중복은 config.yaml에서 읽는다 | PASS | chunk_size=1000, overlap=200 (0.0s) |
| F2006 | 메타데이터 근거가 항상 [근거 1], Top-k를 바꾸면 근거 수가 바뀐다 | PASS | top_k=2 -> 근거 3건, top_k=6 -> 7건 (2.2s) |
| F2006 | 청크 ID·본문 해시·근사 중복 제거 | PASS | 근거 8건, 중복 ID 없음 (1.3s) |
| F2007 | LangGraph가 retrieve -> generate 두 노드 | PASS | 노드 ['retrieve', 'generate'], 엣지 [('__start__', 'retrieve'), ('retrieve', 'generate'), ('generate', '__end__')] (0.0s) |
| F2007 | 문서에 없는 질문: 추측한 수치 없이 자료에 없다고 대화체로 밝힌다 (고정 문구 없음) | PASS | 고정 문구 False, 지어낸 수치 [] / 찾아본 자료에는 **Qwen3-8B 학습에 쓴 총 GPU 시간과 비용**이 나와 있지 않아요. 기술 보고서의 GPU 시간 언급은 증류와 직접 강화학습을 비교하는 (7.4s) |
| F2008 | 실제 인용된 근거만 표시, 인용이 없으면 안내 | PASS | [근거 3][근거 9] -> [3], 없음 안내 '' (9.0s) |
| F2008 | 근거에 없는 수치를 인용 형식으로 쓰면 경고 (환각 검출) | PASS | ['- TopK: 50 [근거 1] (근거에 없는 수치: 50)'] (0.0s) |
| F2008 | 단위가 붙은 소수(180.0B)는 일부(180)만 떼어 검사하지 않는다 (오탐 회귀) | PASS | 경고 [] (0.0s) |
| F2007 | 대화형 답변: 후속 질문은 이전 대화로 이해, 근거 밖 일반 지식에서 근거 없는 수치 문장 제거 | PASS | 후속 '그럼 Q8_0은?' 답에 Q8_0 계산값 9.5 GB 포함=True, 일반 지식 수치 문장 제거 ['보통 수백 GPU 시간이 걸립니다.'] (11.1s) |
| F2007 | 모델과 관계없는 일반 질문(알고리즘 문제)은 문서 검색 없이 일반 답변, 수치 제거 장치를 적용하지 않음 | PASS | 검색어 [], 일반 질문=True, 안내='선택한 모델의 문서와 관계없는 일반 질문이라, 문서 검색 없이 일반 지식', 코드 포함=True (12.3s) |
| F2001 | 최신 멀티모달(image-text-to-text) 모델과 최근 뜨는 원본 모델도 목록에 포함 | PASS | 태그 ['image-text-to-text', 'text-generation'], 포함 ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.6-35B-A3B', 'google/gemma-4-31B-it'], trending 30건 (예: ['deepseek-ai/DeepSeek-V4.1-Flash-1017', 'meta-models/Muse-Spark-Open-816B-A42B', 'Qwen/Qwen3.8-27B']) (0.0s) |
| F2010 | 하이브리드 어텐션 KV: 선형 레이어 제외, sliding 레이어는 창 크기까지만 | PASS | Qwen3.8-27B 32k KV 2.00GB (전체 레이어로 보면 8.00GB, 비율 0.25 = 전체 어텐션 16/64), Gemma4-31B 32k 3.28GB (전체로 보면 30.00GB), 32k→64k 증가분 = 전체 어텐션 레이어 몫 (0.0s) |
| F2010 | 판정·속도 계산 (Qwen3-8B, RTX 3080, 컨텍스트 4096) | PASS | Q4_K_M 5.8 GB GPU에 전부 올라감, 80.1 tok/s (0.0s) |
| F2011 | 조건 탐색 결과가 조건(용도 code)에 맞는 모델만 | PASS | Qwen/Qwen2.5-Coder-14B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct, Qwen/Qwen2.5-Coder-32B-Instruct (14.6s) |
| F2010 | 컨텍스트가 길수록 KV 캐시·필요 메모리는 커지고 속도는 줄어든다 (KV 읽기 포함) | PASS | 2k: KV 0.28GB 85tok/s / 4k: KV 0.56GB 80tok/s / 8k: KV 1.12GB 72tok/s / 16k: KV 2.25GB 61tok/s / 32k: KV 4.50GB 38tok/s / 40k: KV 5.62GB 15tok/s (0.0s) |
| F2010 | 흉내 답변 체급: 유효 파라미터로 구간 결정 (MoE는 sqrt(전체×활성)) | PASS | Qwen3-0.6B=초소형, Qwen3-4B=소형, Qwen3-8B=중형, Qwen3-30B-A3B=중형, Qwen3-32B=대형 (0.0s) |
| F2011 | 추천 상담: 사용자가 말하지 않은 언어 조건은 빼고, 말한 언어는 남긴다 (예시 베끼기 회귀) | PASS | 말 안 함 → '코딩용 모델 내 GPU에서 돌아가는 거' / 말함 → '한국어 되는 코딩용 모델' (0.0s) |
| F2011 | 조건 탐색 검색 문장에서 하드웨어 말을 뺀다 (GPU는 코드 필터로 따로 거름) | PASS | → '수 있는 코딩용 모델' (0.0s) |
| F2011 | 추천 상담: '맥북 M5 16기가'는 사이드바 GPU가 아니라 Apple M5 통합 메모리로 판정 | PASS | ['apple', 'Apple M5', 11.2, 153.0] / 대화에서 알려 주신 하드웨어(Apple M5, GPU 사용 가능 11.2 GB, RAM 16 GB)로 판정합니다. (3.9s) |
| F2011 | 추천 상담: 큰 모델이 안 맞으면 '못 돌림'으로 끝내지 않고 이 PC에 맞는 더 작은 모델을 큰 것부터 제시 | PASS | 27B=일부 CPU 오프로딩 필요, 대안 [('Qwen2.5-Coder-14B-Instruct', 21), ('Qwen2.5-14B-Instruct', 21), ('Qwen3-14B', 21), ('Qwen3.5-9B', 26)] (0.0s) |
| F2011 | 추천 상담: 모델 이름으로 물으면 그 모델을 찾고(띄어쓰기·꼬리 무시), 안 맞으면 같은 계열 대안 | PASS | {'qwen 3.8 flash next 쓸 수 있음?': ['Qwen/Qwen3.8-Flash-Next'], 'gemma 4 31b 어때': ['google/gemma-4-31B-it'], '코딩용 모델 추천해줘': []} / Flash-Next on M5 Max 64GB: 불가, 대안 ['Qwen/Qwen3.8-27B'] (0.0s) |
| F2011 | 추천 상담: 사용자가 말하지 않은 하드웨어는 쓰지 않는다 (LLM이 현재 사양을 되풀이해 40코어 -> 32코어로 바뀌던 회귀) | PASS | 하드웨어 None (6.8s) |
| F2010 | DGX Spark: 통합 메모리 128GB × 0.9(가정값)를 GPU 몫으로, 넘치면 같은 메모리에서 오프로딩, 대역폭 273GB/s | PASS | unified, 메모리 128.0 GB, GPU 몫 115.2 GB, 273.0 GB/s, Flash-Next FP16 369 GB → 불가 (0.0s) |
| F2011 | 대화 속 GPU 이름 매칭 (줄임말·VRAM 구분) | PASS | {'4090': 'RTX 4090', '3080 ti': 'RTX 3080 Ti', 'RTX 3080': 'RTX 3080', 'M4 Pro': 'Apple M4 Pro'} (0.0s) |
| F2012 | 라이선스 질문에 면책 안내 | PASS | 출처 표시도 유지해야 합니다. [근거 2] 법률 자문이 아니며, 정확한 조건은 원문 라이선스를 확인하세요. (12.8s) |
| NF2001 | 대표 질문 응답 시간 (수집 캐시 히트 상태) | PASS | 6.8초 (단계별 수집 1.7s, 검색 1.9s, 생성 3.2s) (6.8s) |
| NF2003 | API 키가 커밋된 파일에 없다 | PASS | 추적 파일 98개 중 키 포함 0개, .env gitignore=True (0.1s) |
| NF2004 | 오류는 사용자 문구로 바꾸고 스택 트레이스를 노출하지 않는다 | PASS | 답변을 만드는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. (0.0s) |
| NF2005 | 수집·색인·검색·생성·UI 모듈 분리, 도메인 의존 코드는 data_prep | PASS | app/data_prep, app/collect, app/index, app/rag, app/features, app/ui (0.0s) |
| NF2006 | User-Agent 연락처, 호스트별 호출 간격(arXiv 3초), 재시도 | PASS | can-i-run-this/0.1.0 (https://github.com/kbgkbg/can-i-run-this) python-requests/2.34.2 (0.0s) |
| NF2007 | 데이터·문서 출처와 이용 조건 기록 | PASS | data/SOURCES.md (0.0s) |
