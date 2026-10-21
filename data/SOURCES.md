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
| Hugging Face 라이선스 원문 (라이선스 질문일 때만) | `https://huggingface.co/{모델ID}/raw/main/{license_link 경로 또는 LICENSE, LICENSE.md, LICENSE.txt}` | 파일 URL | 라이선스 원문 자체. 답변에는 "법률 자문이 아니며, 정확한 조건은 원문 라이선스를 확인하세요." 안내를 붙인다 |
| 위키백과 (한국어 우선, 없으면 영어) | MediaWiki API `list=search`, `prop=extracts&explaintext` | 문서 URL | CC BY-SA 4.0 — 출처(문서 URL)와 라이선스를 표시. User-Agent에 프로젝트명과 연락처 포함 |
| arXiv PDF (Phase 8) | | PDF URL + 쪽 번호 | |

- 수집 시각은 청크 메타데이터 `collected_at`에 문서별로 남는다.
- 원문은 `data/cache/docs/`에 저장되고(커밋하지 않음), 같은 문서를 다른 모델이 고르면 다시 받지 않는다.
- `data/offline_docs/`: 외부 API를 쓸 수 없는 환경용 예시 묶음 (`scripts/export_offline_docs.py`로 생성, 2026-10-21 수집분 6건).
  각 파일에 원문 URL과 라이선스(`license` 필드)가 들어 있다.

## 3. 하드웨어 사양 표 — `data/hw_specs.csv`

수치는 제조사 공식 자료에서 직접 받아 확인한 값만 넣었다 (2026-10-21). 행마다 `source_url`과 `note`(표·쪽 위치)를 둔다.
공식 자료에서 확인하지 못한 칸은 `TODO`로 두고, 판정·속도 추정에서 "정보 없음"으로 처리한다.

| 출처 | 사용한 값 |
|---|---|
| NVIDIA GeForce 비교표 https://www.nvidia.com/en-us/geforce/graphics-cards/compare/ | RTX 50 시리즈 VRAM·메모리 대역폭, RTX 40/30 시리즈 VRAM |
| NVIDIA Ada GPU Architecture 백서 (Table 2, 3) https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf | RTX 4090, 4080, 3090 Ti, 3080 Ti 대역폭 |
| NVIDIA Ampere GA102 백서 (Table 2, 9, 10) https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf | RTX 3090, 3080(10 GB), 3070 대역폭 |
| Apple 기술 사양 페이지 (support.apple.com/en-us/111901, 111900, 111867, 111340, 111835, 117735, 117736, 121552, 121553) | Apple M1 Pro/Max/Ultra ~ M4 Pro/Max 메모리 대역폭 |

**TODO (공식 출처 미확인):** RTX 4080 SUPER, 4070 Ti SUPER, 4070 Ti, 4070 SUPER, 4070, 4060 Ti, 4060, 3080 12GB, 3070 Ti, 3060 Ti, 3060의 대역폭,
Apple M1(기본)의 대역폭 (사양 페이지 111893에 표기 없음).

## 4. 계산에 쓰는 그 밖의 값 — `app/config.yaml`

| 값 | 출처 / 성격 |
|---|---|
| 양자화 bpw Q6_K 6.5625, Q8_0 8.5 | llama.cpp `ggml/src/ggml-common.h` 블록 구조 정의 (MIT 라이선스) |
| 양자화 bpw Q4_K_M 4.97, Q5_K_M 5.75 | 공식 Qwen3-4B/8B/14B/32B-GGUF 파일 크기 실측의 최댓값 (`scripts/measure_bpw.py`) |
| overhead_ratio, speed_efficiency, ram_usable_ratio, apple_gpu_mem_ratio, stream_chars_per_token | 가정값 (실측으로 보정 예정) |
| system_ram_bandwidth_gbps 51.2 | DDR4-3200 듀얼채널 규격 정의에 따른 계산 (3200 MT/s × 8바이트 × 2), 사용자 PC에 맞게 조정 |
