# Can I run this model?

> 로컬 LLM을 내 PC에서 돌릴 수 있는지, 얼마나 빠른지, 어떤 양자화를 써야 하는지를
> **모델 메타데이터와 외부 문서(Hugging Face 모델 카드·라이선스 원문, 위키백과, arXiv 기술 보고서)를 근거로** 답하는 RAG 지식 서비스
> (SSAFY AI 코어 프로젝트 2회차)

- 사용자는 모델을 골라 질문하고, 서비스는 근거 문서를 수집·검색해 답하며 **답변에 실제로 인용한 근거만** 원문 링크와 함께 보여 준다.
- 서버(Streamlit)가 사용자 PC에서 돌기 때문에 **PC 하드웨어를 직접 읽어** 실행 가능 여부와 토큰 속도를 계산한다.
- 배포하지 않고 로컬에서만 실행한다.

| 문서 | 내용 |
|---|---|
| [docs/01_requirements.md](docs/01_requirements.md) | 요구사항 정의서: 도메인 조건 충족 근거, 필수·심화 요구사항 표 |
| [docs/02_rag_design.md](docs/02_rag_design.md) | RAG 파이프라인 설계문서: 컬렉션, 청킹, 프롬프트, Top-k, 겪은 문제와 해결 |
| [docs/03_verification.md](docs/03_verification.md) | 검증 결과: 요구사항 대조표, 테스트 결과표, 상호 체크리스트 |
| [docs/eval/](docs/eval/) | 자동 점검 결과, MMR 비교, RAG 전후 비교 |
| [data/SOURCES.md](data/SOURCES.md) | 데이터·문서 출처와 이용 조건 |

---

## 1. 실행 방법

Python 3.10 이상 (개발·검증: 3.13).

```bash
# 1) 설치
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 환경변수: .env.example을 .env로 복사하고 값을 채운다 (.env는 커밋하지 않는다)
cp .env.example .env
#   OPENAI_API_KEY   SSAFY GMS 키 (필수)
#   OPENAI_BASE_URL  https://gms.ssafy.io/gmsapi/api.openai.com/v1 (필수)
#   CONTACT_INFO     외부 API User-Agent에 넣을 연락처: 이메일 또는 저장소 URL (필수, 위키백과 정책)
#   HF_TOKEN         선택. gated 모델(Llama, Gemma 등)의 config.json·모델 카드를 받을 때
#   WIKI_ACCESS_TOKEN 선택. 위키백과 OAuth 2.0 토큰 (IP 기준 분당 10회 한도를 계정 기준 200회로)

# 3) 모델 메타데이터 색인 (data/models.json 스냅샷이 저장소에 있으므로 바로 색인된다)
python scripts/build_meta_index.py            # --fetch: HF Hub에서 새로 수집한 뒤 색인

# 4) 실행
streamlit run app/ui/streamlit_app.py
```

> **LLM 설정** (`app/config.yaml llm`): 답변·흉내 대화 `gpt-6-luna`(SSAFY GMS, temperature는 기본값만 허용해 보내지 않음), 임베딩 `text-embedding-3-large`.
> 임베딩 모델을 바꾸면 기존 색인과 차원이 달라 `python scripts/build_meta_index.py`를 다시 실행해야 한다 (수집해 둔 문서까지 자동으로 다시 색인, 재수집 없음).

### 화면 사용 순서
1. 사이드바에서 **모델 선택** (이름 입력으로 검색), **🔎 조건으로 모델 찾기** ("한국어 되는 코딩용 모델, 내 그래픽카드에서 돌아가는 거"),
   또는 모드를 **추천 상담**으로 바꿔 챗봇과 대화하며 추천받기 ("코딩용이면 돼. 그래픽카드는 3080, 램 32기가")
2. 처음 열면 **내 PC 사양 선택** 화면이 나온다: 자동 감지(NVIDIA·Apple Silicon) 또는 탭에서 직접 선택
   — **🍎 Mac**(칩·통합 메모리·GPU 사용 비율) / **🟩 NVIDIA**(GPU·시스템 RAM) / **💻 GPU 없음**(RAM). 사이드바에서 언제든 변경
3. 판정 카드에서 양자화별 필요 메모리·판정·추정 속도 확인, 컨텍스트 길이 조절, **📏 컨텍스트 길이별 차이** 표 (2k ~ 모델 최대)
4. 채팅으로 질문 (화면 아래 💬 입력창, 또는 안내 말풍선의 예시 질문 버튼). 같은 모델과의 대화를 이어 받아 "그럼 Q5는?" 같은 후속 질문도 이해한다.
   근거로 확인되는 내용은 [근거 n]으로, 문서에 없는 부분은 "💡 일반적인 경우 (근거 문서 아님)"으로 구분해 답한다.
   ⏳ 모델마다 **첫 질문은 근거 문서를 모으느라 수십 초** 걸릴 수 있고(측정 예: Qwen3.8-Flash-Next 첫 질문 약 18초, 두 번째 약 9초), 다음 질문부터 빨라진다.
   **내 PC 속도로 체험**을 (사양을 아직 안 골랐으면 켤 때 이 PC를 자동 감지한다. 대역폭이 사양표에 없는 GPU·칩은 사양 선택에서 직접 입력하면 된다) 켜면 답변이 추정 tok/s 속도로 흘러나온다 (오프로딩 판정이어도 GPU·RAM 분담 속도로 재생)
5. **🎭 이 모델과 대화해 보기**를 켜면 문서 검색 없이 그 모델 체급의 로컬 챗봇을 연기해 아무 주제나 대화한다(상식·계산·코딩으로 체급 차이를 체감). 이전 대화를 기억하고 근거 Q&A 기록과 따로 관리한다.
   이전 설명: 근거 답변 아래에 그 모델 체급이 근거 없이 답했다면 어땠을지 연기한 답이 붙는다
   (켜면 근거 답변 대신 🎭 연기 답변만 나온다. 실제 모델 출력 아님. 호스팅 추론 비용 없이 같은 LLM으로 연기한다)

### 검증·실험 스크립트
| 명령 | 용도 |
|---|---|
| `python scripts/run_checks.py` | 요구사항 자동 점검 31항목 → `docs/eval/test_results.md` |
| `python scripts/ui_checks.py` | 화면 점검 12항목 (첫 화면 사양 선택, 빈 대화 안내·예시 질문, 속도 체험 자동 감지·대역폭 직접 입력, 모델별 대화, 진행 상태, 키 누락·색인 0건·잘못된 키, 컨텍스트 전환 회귀, 흉내 답변, 추천 상담) → `docs/eval/ui_test_results.md` |
| `python scripts/collect_demo.py "<모델ID>" "<질문>"` | 프롬프트 1 검색어, 후보, 프롬프트 2 선별/제외, 색인 결과, 캐시 사용 여부 |
| `python scripts/ask_demo.py "<모델ID>" "<질문>" [--top-k N] [--search-type mmr]` | 근거 목록, 답변, 인용 근거, 수치 검증 경고 |
| `python scripts/reindex_docs.py` | 수집한 원문을 현재 청크 설정으로 재색인 (청크 크기 변경 효과 비교) |
| `python scripts/compare_mmr.py` | 유사도 vs MMR 근거 비교 → `docs/eval/mmr_comparison.md` |
| `python scripts/compare_rag.py` | RAG 사용 전후 답변 비교 → `docs/eval/rag_comparison.md` |
| `python scripts/recommend_demo.py "<조건>" [--gpu "RTX 3080"]` | 조건 기반 모델 탐색 |
| `python scripts/measure_bpw.py` | 공식 GGUF 파일 크기로 양자화 실효 bpw 측정 |
| `python scripts/export_offline_docs.py` | 수집 문서를 오프라인 묶음(`data/offline_docs/`)으로 내보내기 |
| `python -m app.features.hw_detect` | 하드웨어 감지 결과 JSON 출력 |
| `python -m app.rag.graph` | LangGraph 구조 출력 (retrieve → generate) |

외부 API를 쓸 수 없는 환경에서는 `app/config.yaml`의 `collect.offline_mode: true`로 두면 `data/offline_docs/` 묶음에서 수집한다.

---

## 2. 주요 기능

### 필수 기능 (F2001 ~ F2012)
| 번호 | 기능 | 구현 |
|---|---|---|
| F2001 | HF Hub text-generation·image-text-to-text(최신 멀티모달) 모델 150건(최근 뜨는 원본 모델 30건 포함) 메타데이터(파라미터·레이어·KV 헤드·컨텍스트·라이선스·언어·태그)를 항목 텍스트 1개로 | `app/data_prep/fetch_models.py`, `item_text.py` |
| F2002 | `model_meta` 컬렉션, 항목당 1청크, 청크 ID = 모델 ID (재색인해도 건수 불변) | `app/index/`, `scripts/build_meta_index.py` |
| F2003 | 프롬프트 1(로컬 LLM 배포 전문가): 질문 → 검색어 JSON 배열 | `app/rag/prompts.py`, `app/collect/collector.py` |
| F2004 | 출처별 후보(제목+요약) → 프롬프트 2 선별 → 본문 수집. 선택 모델의 모델 카드는 항상 포함, 수집 캐시 | `app/collect/` |
| F2005 | `model_docs` 컬렉션, 출처별 청킹(모델 카드는 마크다운 헤더 기준, PDF는 쪽 단위), 청크 ID = 모델ID:문서ID:순번 | `app/index/indexer.py` |
| F2006 | 메타데이터 [근거 1] 고정 + 모델 필터 검색 + 개념 문서 검색 병합, 청크 ID·해시·근사 중복 제거, Top-k 설정 | `app/rag/retrieve.py` |
| F2007 | LangGraph retrieve → generate, 프롬프트 3(근거 우선 상담형 답변. 근거에 없는 부분은 대화체로 밝히고 추측한 수치는 쓰지 않음) | `app/rag/graph.py` |
| F2008 | [근거 n] 파싱, 실제 인용 근거만 제목·출처 유형·원문 링크로 표시, 없으면 안내 | `app/rag/citations.py` |
| F2009 | 모델 선택, 모델별로 분리된 대화 | `app/ui/streamlit_app.py` |
| F2010 | **도메인 특화**: 하드웨어 감지, 실행 가능 판정(4단계), 토큰 속도 추정·체험, 라이선스 원문 안내. 판정에도 근거([근거 1]·사양표 출처)와 "계산값" 구분 | `app/features/` |
| F2011 | 자연어 조건 → 후보 검색 → 코드 필터 + LLM 선별(인용 근거 검증) → 판정이 붙은 카드, 카드 클릭 시 대화로. **추천 상담 모드**: 챗봇이 조건·하드웨어를 대화로 모아 같은 탐색 호출 | `app/features/recommend.py`, `consult.py` |
| F2012 | 진행 상태(검색어 추출 → 후보 선별 → 수집 → 색인 → 검색 → 생성), 키 누락·색인 0건·외부 API 실패 안내, 라이선스 면책 | `app/ui/`, `app/rag/pipeline.py` |

### 심화 기능
| 번호 | 기능 | 결과 |
|---|---|---|
| F2014 | MMR 검색 (기본값) | 4개 질문 모두 근거 간 평균 유사도가 유사도 검색보다 낮음 → [mmr_comparison.md](docs/eval/mmr_comparison.md) |
| F2015 | RAG 사용 전후 비교 (대표 질문 7개) | 사실 정확성 0/5 → 5/5, 근거 제시 0/7 → 6/7 → [rag_comparison.md](docs/eval/rag_comparison.md) |
| F2016 | 여러 출처 결합: HF 모델 카드 + 라이선스 원문 + 위키백과 + arXiv PDF(쪽 번호) | 근거 라벨 `HF 모델 카드` / `라이선스 원문` / `위키백과` / `arXiv PDF p.3` |
| (추가) | 근거 수치 검증: [근거 n]을 붙인 줄의 수치가 그 근거에 실제로 있는지 검사해 경고 | 근거에 없는 값을 인용 형식으로 지어낸 사례 대응 |
| (추가) | 🎭 모델 흉내 답변: 유효 파라미터 체급별로 정해진 수의 실수를 넣어 연기, 속도 체험과 함께 재생 | 0.6B는 레이어·컨텍스트를 틀리고 문장 반복, 32B는 그대로 → 체급 차이 체감 |
| F2013 | HTML/JS 프론트엔드 직접 구현 | 선택하지 않음 (Streamlit) |

---

## 3. 데이터 준비 및 색인 절차

1. `python -m app.data_prep.fetch_models` — HF Hub에서 text-generation·image-text-to-text 모델을 다운로드순으로, trendingScore 상위 원본 모델 30건을 먼저 받아 양자화 파생·테스트용 저장소를 거르고 150건의 `config.json`(레이어별 어텐션 종류 포함)까지 읽어 `data/models.json`에 저장 (저장소에 2026-10-22 스냅샷 포함). Qwen3.5 이후 Qwen, Gemma 4처럼 최신 모델은 멀티모달이라 image-text-to-text로만 분류된다
2. `python scripts/build_meta_index.py` — 항목 텍스트를 임베딩해 `model_meta`에 upsert
3. 외부 문서는 **질문할 때** 수집된다 (`model_docs`). 수집한 원문은 `data/cache/docs/`에 저장되어 같은 문서를 다시 받지 않고, 청크 설정을 바꾸면 `scripts/reindex_docs.py`로 재색인한다
4. 하드웨어 사양표 `data/hw_specs.csv`는 제조사 공식 자료에서 확인한 값만 넣었다 (미확인 칸은 `TODO`)

모든 조정값(청크 크기·중복, Top-k, 검색 방식, MMR 파라미터, 모델명, 양자화 bpw, 효율·오버헤드 계수, 호출 간격)은 `app/config.yaml`에 있다.

## 4. 데이터 출처와 이용 조건

자세한 내용은 [data/SOURCES.md](data/SOURCES.md).

| 데이터 | 출처 | 이용 조건 |
|---|---|---|
| 모델 메타데이터 | Hugging Face Hub API, 각 모델 `config.json` | HF 서비스 약관. 모델 라이선스는 항목별 `license` |
| 모델 카드 / 라이선스 원문 | 각 모델 저장소 `README.md`, `LICENSE` | 모델별 라이선스. 원문 링크 표시 |
| 위키백과 | MediaWiki API (한국어 우선) | CC BY-SA 4.0, User-Agent에 연락처 |
| arXiv 기술 보고서 | arXiv API, PDF | 논문별 라이선스, 호출 간격 3초. 오프라인 묶음에는 넣지 않음 |
| GPU·Apple 칩 사양 | NVIDIA 비교표·Ada/Ampere 백서, Apple 기술 사양 페이지 | 출처 URL을 행마다 기록 |
| 양자화 bpw | llama.cpp `ggml-common.h`, 공식 GGUF 파일 크기 실측 | MIT / 측정값 |

## 5. 프로젝트 구조

```
app/
  config.yaml  config.py  log.py
  data_prep/   fetch_models.py  item_text.py  hw_tables.py     # 도메인에 묶인 부분 (NF2005)
  collect/     http.py  cache.py  collector.py
               sources/  hf_card.py  wikipedia.py  arxiv_pdf.py  offline.py  base.py
  index/       store.py  indexer.py
  rag/         prompts.py  llm.py  retrieve.py  graph.py  citations.py  intent.py  pipeline.py
  features/    hw_detect.py  fit.py  speed.py  recommend.py
  ui/          streamlit_app.py
scripts/       색인·검증·실험 스크립트
data/          models.json  hw_specs.csv  SOURCES.md  offline_docs/
docs/          요구사항 정의서, 설계문서, 검증 결과, eval/, screenshots/
```

## 6. 알려진 한계

- 🎭 흉내 답변은 **연기**다. 체급 구간과 실수 개수는 가정값이며 실제 모델의 지능·말투를 재현하지 않는다.
- "💡 일반적인 경우 (근거 문서 아님)" 부분은 근거 검증을 하지 않는 일반 지식이다. 근거 없는 수치 문장은 코드로 빼지만 서술 자체가 틀릴 수 있다.
- 멀티모달 모델은 이미지 입력용 비전 인코더까지 파라미터 수에 포함해 계산한다. 선형 어텐션 레이어의 고정 상태 메모리는 0으로 둔다.
- 판정·속도는 공식으로 낸 **추정치(계산값)** 다. 효율·오버헤드 계수는 가정값이며 실측으로 보정해야 한다.
- RTX 40 시리즈 대부분과 Apple M1의 메모리 대역폭은 공식 출처를 찾지 못해 `TODO` → 속도 추정을 생략한다.
- gated 모델(Llama, Gemma 등)은 `HF_TOKEN` 없이는 레이어 수·모델 카드를 받을 수 없어 "정보 부족"으로 나온다.
- 언어 조건 탐색은 모델 카드 메타데이터에 언어가 표기된 모델만 확인할 수 있다.
- 공유 IP 환경에서는 위키백과가 429(분당 10회 한도)를 낼 수 있다 → `WIKI_ACCESS_TOKEN`.
- 명세서(`docs/spec.pdf`)는 對外秘 문서라 저장소에 포함하지 않는다.
