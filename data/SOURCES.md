# 데이터·문서 출처와 이용 조건 (NF2007)

## 1. 모델 메타데이터 — `data/models.json`

| 항목 | 내용 |
|---|---|
| 출처 | Hugging Face Hub API `https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads` |
| 세부 필드 | 레이어 수·KV 헤드·컨텍스트 길이: 각 모델 저장소의 `config.json` (`https://huggingface.co/{모델ID}/resolve/main/config.json`) |
| 파라미터 수 | Hub API의 `safetensors.total` (safetensors 메타데이터 기준) |
| 수집일 | 2026-10-21 (UTC 07:06, `models.json`의 `collected_at`) |
| 건수 | 다운로드순 목록 400건 중 150건 저장 |
| 제외 기준 | 양자화 파생 저장소(이름에 GGUF/AWQ/GPTQ/FP8 등) 185건, safetensors 메타데이터 없음 11건, 테스트용 저장소 13건, 5천만 파라미터 미만 3건 |
| 정보 부족 | gated 모델(Meta Llama, Google Gemma 등) 14건은 토큰 없이 `config.json`을 받을 수 없어 레이어 수 등이 `null` → 판정 시 "정보 부족" |
| 이용 조건 | Hugging Face 서비스 약관(https://huggingface.co/terms-of-service)에 따라 공개 API로 조회한 메타데이터. 모델 자체의 라이선스는 항목별 `license` 필드와 각 모델 페이지 원문을 따른다 |
| 재수집 | `python -m app.data_prep.fetch_models` |

## 2. 외부 문서 (질문 시점 수집, `model_docs` 컬렉션)

| 출처 | 수집 방법 | 원문 출처 표기 | 이용 조건 |
|---|---|---|---|
| Hugging Face 모델 카드 | `https://huggingface.co/{모델ID}/raw/main/README.md` (후보 검색: `/api/models?search=`) | 모델 페이지 URL | 모델 카드별 라이선스(각 저장소의 license 필드·LICENSE 파일)를 따른다. 원문 링크를 함께 표시 |
| 위키백과 (한국어 우선, 없으면 영어) | MediaWiki API `list=search`, `prop=extracts&explaintext` | 문서 URL | CC BY-SA 4.0 — 출처(문서 URL)와 라이선스를 표시. User-Agent에 프로젝트명과 연락처 포함 |
| arXiv PDF (Phase 8) | | PDF URL + 쪽 번호 | |

- 수집 시각은 청크 메타데이터 `collected_at`에 문서별로 남는다.
- 원문은 `data/cache/docs/`에 저장되고(커밋하지 않음), 같은 문서를 다른 모델이 고르면 다시 받지 않는다.
- `data/offline_docs/`: 외부 API를 쓸 수 없는 환경용 예시 묶음 (`scripts/export_offline_docs.py`로 생성, 2026-10-21 수집분 6건).
  각 파일에 원문 URL과 라이선스(`license` 필드)가 들어 있다.

## 3. 하드웨어 사양 표 — `data/hw_specs.csv`

Phase 6에서 채움: 제조사 공식 사양 페이지를 행마다 `source_url`로 기록
