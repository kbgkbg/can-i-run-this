# 유사도 검색 vs MMR 검색 비교 (F2014)

- top_k=8, MMR fetch_k=30, lambda=0.5, 근사 중복 기준 자카드 0.8, 청크 1000/200자
- 지표는 메타데이터 근거([근거 1])를 뺀 외부 문서 근거 기준. 평균 코사인 유사도는 낮을수록 근거끼리 덜 겹친다.
- 생성: `python scripts/compare_mmr.py`

## 요약

| 질문 | A. similarity (중복 제거 끔) | B. similarity + 근사 중복 제거 | C. MMR + 근사 중복 제거 (기본) |
|---|---|---|---|
| Qwen3-8B의 구조(레이어, 어텐션 헤드)와 컨텍스트 길이는? | 문서 3종 · 유사도 0.729 | 문서 2종 · 유사도 0.769 | 문서 4종 · 유사도 0.540 |
| Qwen3는 사전 학습에 어떤 데이터를 얼마나 썼어? | 문서 3종 · 유사도 0.723 | 문서 3종 · 유사도 0.726 | 문서 2종 · 유사도 0.607 |
| thinking 모드와 non-thinking 모드의 권장 샘플링 설정은? | 문서 2종 · 유사도 0.571 | 문서 2종 · 유사도 0.571 | 문서 2종 · 유사도 0.531 |
| 긴 텍스트를 처리하려면 어떻게 설정해야 해? | 문서 3종 · 유사도 0.480 | 문서 3종 · 유사도 0.480 | 문서 3종 · 유사도 0.374 |

## Qwen/Qwen3-8B — Qwen3-8B의 구조(레이어, 어텐션 헤드)와 컨텍스트 길이는?

**A. similarity (중복 제거 끔)** — 서로 다른 문서 3종, 근거 간 평균 코사인 유사도 0.729

- [근거 2] (arXiv PDF p.29) Qwen3 Technical Report (arXiv:2505.09388) · `124번 청크` · Qwen3-235B-A22B 61.3 23.3 27.6 37.4
- [근거 3] (arXiv PDF p.28) Qwen3 Technical Report (arXiv:2505.09388) · `121번 청크` · Qwen2.5-72B-Instruct 61.3 72.7 26.7 18.6 44.8
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `2번 청크` · [Qwen3-8B > Model Overview]
- [근거 5] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `2번 청크` · [Qwen3-8B-GGUF > Model Overview]
- [근거 6] (arXiv PDF p.27) Qwen3 Technical Report (arXiv:2505.09388) · `118번 청크` · Qwen3-0.6B 36.4 46.6 10.0 12.6 26.4
- [근거 7] (arXiv PDF p.26) Qwen3 Technical Report (arXiv:2505.09388) · `115번 청크` · Qwen3-4B 63.7 80.4 72.5 53.3 40.7 62.1
- [근거 8] (arXiv PDF p.24) Qwen3 Technical Report (arXiv:2505.09388) · `109번 청크` · Qwen3-14B 77.6 71.2 73.8 80.4 73.3 44.2 70.1
- [근거 9] (arXiv PDF p.9) Qwen3 Technical Report (arXiv:2505.09388) · `38번 청크` · Qwen3-30B-A3B-Base. (2) Qwen2.5-32B-Base, which has 11 times the activ

**B. similarity + 근사 중복 제거** — 서로 다른 문서 2종, 근거 간 평균 코사인 유사도 0.769

- [근거 2] (arXiv PDF p.29) Qwen3 Technical Report (arXiv:2505.09388) · `124번 청크` · Qwen3-235B-A22B 61.3 23.3 27.6 37.4
- [근거 3] (arXiv PDF p.28) Qwen3 Technical Report (arXiv:2505.09388) · `121번 청크` · Qwen2.5-72B-Instruct 61.3 72.7 26.7 18.6 44.8
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `2번 청크` · [Qwen3-8B > Model Overview]
- [근거 5] (arXiv PDF p.27) Qwen3 Technical Report (arXiv:2505.09388) · `118번 청크` · Qwen3-0.6B 36.4 46.6 10.0 12.6 26.4
- [근거 6] (arXiv PDF p.26) Qwen3 Technical Report (arXiv:2505.09388) · `115번 청크` · Qwen3-4B 63.7 80.4 72.5 53.3 40.7 62.1
- [근거 7] (arXiv PDF p.24) Qwen3 Technical Report (arXiv:2505.09388) · `109번 청크` · Qwen3-14B 77.6 71.2 73.8 80.4 73.3 44.2 70.1
- [근거 8] (arXiv PDF p.9) Qwen3 Technical Report (arXiv:2505.09388) · `38번 청크` · Qwen3-30B-A3B-Base. (2) Qwen2.5-32B-Base, which has 11 times the activ
- [근거 9] (arXiv PDF p.25) Qwen3 Technical Report (arXiv:2505.09388) · `112번 청크` · Qwen3-30B-A3B 75.9 94.5 81.9 80.0 48.1 76.1

**C. MMR + 근사 중복 제거 (기본)** — 서로 다른 문서 4종, 근거 간 평균 코사인 유사도 0.540

- [근거 2] (arXiv PDF p.29) Qwen3 Technical Report (arXiv:2505.09388) · `124번 청크` · Qwen3-235B-A22B 61.3 23.3 27.6 37.4
- [근거 3] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `2번 청크` · [Qwen3-8B > Model Overview]
- [근거 4] (arXiv PDF p.1) Qwen3 Technical Report (arXiv:2505.09388) · `2번 청크` · etc., competitive against larger MoE models and proprietary models. Co
- [근거 5] (arXiv PDF p.15) Qwen3 Technical Report (arXiv:2505.09388) · `73번 청크` · Qwen3-30B-A3B & Qwen3-14B For Qwen3-30B-A3B and Qwen3-14B, we compare 
- [근거 6] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `6번 청크` · [Qwen3-8B-GGUF > Processing Long Texts]
- [근거 7] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `1번 청크` · [Qwen3-8B > Qwen3 Highlights]
- [근거 8] (arXiv PDF p.19) Qwen3 Technical Report (arXiv:2505.09388) · `86번 청크` · AIME’25 22.8 27.8 15.1 36.8
- [근거 9] (HF 모델 카드) Aldaris/Qwen3-8B-Q4_K_M-GGUF 모델 카드 · `0번 청크` · [Aldaris/Qwen3-8B-Q4_K_M-GGUF]

## Qwen/Qwen3-8B — Qwen3는 사전 학습에 어떤 데이터를 얼마나 썼어?

**A. similarity (중복 제거 끔)** — 서로 다른 문서 3종, 근거 간 평균 코사인 유사도 0.723

- [근거 2] (arXiv PDF p.3) Qwen3 Technical Report (arXiv:2505.09388) · `14번 청크` · our training data. Specifically, we collected twice as many pre-traini
- [근거 3] (arXiv PDF p.3) Qwen3 Technical Report (arXiv:2505.09388) · `13번 청크` · Table 1: Model architecture of Qwen3 dense models.
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `2번 청크` · [Qwen3-8B-GGUF > Model Overview]
- [근거 5] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `2번 청크` · [Qwen3-8B > Model Overview]
- [근거 6] (arXiv PDF p.4) Qwen3 Technical Report (arXiv:2505.09388) · `17번 청크` · studies (Xie et al., 2023; Fan et al., 2023; Liu et al., 2024b) that o
- [근거 7] (arXiv PDF p.10) Qwen3 Technical Report (arXiv:2505.09388) · `40번 청크` · The post-training pipeline of Qwen3 is strategically designed with two
- [근거 8] (arXiv PDF p.1) Qwen3 Technical Report (arXiv:2505.09388) · `0번 청크` · 2025-05-15
- [근거 9] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `0번 청크` · [Qwen3-8B > Qwen3 Highlights]

**B. similarity + 근사 중복 제거** — 서로 다른 문서 3종, 근거 간 평균 코사인 유사도 0.726

- [근거 2] (arXiv PDF p.3) Qwen3 Technical Report (arXiv:2505.09388) · `14번 청크` · our training data. Specifically, we collected twice as many pre-traini
- [근거 3] (arXiv PDF p.3) Qwen3 Technical Report (arXiv:2505.09388) · `13번 청크` · Table 1: Model architecture of Qwen3 dense models.
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `2번 청크` · [Qwen3-8B-GGUF > Model Overview]
- [근거 5] (arXiv PDF p.4) Qwen3 Technical Report (arXiv:2505.09388) · `17번 청크` · studies (Xie et al., 2023; Fan et al., 2023; Liu et al., 2024b) that o
- [근거 6] (arXiv PDF p.10) Qwen3 Technical Report (arXiv:2505.09388) · `40번 청크` · The post-training pipeline of Qwen3 is strategically designed with two
- [근거 7] (arXiv PDF p.1) Qwen3 Technical Report (arXiv:2505.09388) · `0번 청크` · 2025-05-15
- [근거 8] (HF 모델 카드) Qwen/Qwen3-8B 모델 카드 · `0번 청크` · [Qwen3-8B > Qwen3 Highlights]
- [근거 9] (arXiv PDF p.2) Qwen3 Technical Report (arXiv:2505.09388) · `6번 청크` · viding users with fine-grained control over the level of reasoning eff

**C. MMR + 근사 중복 제거 (기본)** — 서로 다른 문서 2종, 근거 간 평균 코사인 유사도 0.607

- [근거 2] (arXiv PDF p.3) Qwen3 Technical Report (arXiv:2505.09388) · `14번 청크` · our training data. Specifically, we collected twice as many pre-traini
- [근거 3] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `2번 청크` · [Qwen3-8B-GGUF > Model Overview]
- [근거 4] (arXiv PDF p.4) Qwen3 Technical Report (arXiv:2505.09388) · `17번 청크` · studies (Xie et al., 2023; Fan et al., 2023; Liu et al., 2024b) that o
- [근거 5] (arXiv PDF p.3) Qwen3 Technical Report (arXiv:2505.09388) · `11번 청크` · 2 Architecture
- [근거 6] (arXiv PDF p.9) Qwen3 Technical Report (arXiv:2505.09388) · `39번 청크` · Gemma-3 base models as the baselines. The results can be seen in Table
- [근거 7] (arXiv PDF p.22) Qwen3 Technical Report (arXiv:2505.09388) · `97번 청크` · (2) Stage 4 further strengthens the model’s general, instruction-follo
- [근거 8] (arXiv PDF p.15) Qwen3 Technical Report (arXiv:2505.09388) · `73번 청크` · Qwen3-30B-A3B & Qwen3-14B For Qwen3-30B-A3B and Qwen3-14B, we compare 
- [근거 9] (arXiv PDF p.24) Qwen3 Technical Report (arXiv:2505.09388) · `109번 청크` · Qwen3-14B 77.6 71.2 73.8 80.4 73.3 44.2 70.1

## Qwen/Qwen3-8B — thinking 모드와 non-thinking 모드의 권장 샘플링 설정은?

**A. similarity (중복 제거 끔)** — 서로 다른 문서 2종, 근거 간 평균 코사인 유사도 0.571

- [근거 2] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `8번 청크` · [Qwen3-8B-GGUF > Best Practices]
- [근거 3] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `49번 청크` · Chat Template Design. To better integrate the two modes and enable use
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `5번 청크` · [Qwen3-8B-GGUF > Switching Between Thinking and Non-Thinking Mode]
- [근거 5] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `50번 청크` · think block in the chat template. By default, the model operates in th
- [근거 6] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `51번 청크` · model’s thinking reaches a user-defined threshold, we manually halt th
- [근거 7] (arXiv PDF p.13) Qwen3 Technical Report (arXiv:2505.09388) · `63번 청크` · languages, sourced from Zhang et al. (2024).
- [근거 8] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `46번 청크` · Table 9: Examples of SFT data for thinking and non-thinking modes duri
- [근거 9] (arXiv PDF p.12) Qwen3 Technical Report (arXiv:2505.09388) · `57번 청크` · 4.6 Post-training Evaluation

**B. similarity + 근사 중복 제거** — 서로 다른 문서 2종, 근거 간 평균 코사인 유사도 0.571

- [근거 2] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `8번 청크` · [Qwen3-8B-GGUF > Best Practices]
- [근거 3] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `49번 청크` · Chat Template Design. To better integrate the two modes and enable use
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `5번 청크` · [Qwen3-8B-GGUF > Switching Between Thinking and Non-Thinking Mode]
- [근거 5] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `50번 청크` · think block in the chat template. By default, the model operates in th
- [근거 6] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `51번 청크` · model’s thinking reaches a user-defined threshold, we manually halt th
- [근거 7] (arXiv PDF p.13) Qwen3 Technical Report (arXiv:2505.09388) · `63번 청크` · languages, sourced from Zhang et al. (2024).
- [근거 8] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `46번 청크` · Table 9: Examples of SFT data for thinking and non-thinking modes duri
- [근거 9] (arXiv PDF p.12) Qwen3 Technical Report (arXiv:2505.09388) · `57번 청크` · 4.6 Post-training Evaluation

**C. MMR + 근사 중복 제거 (기본)** — 서로 다른 문서 2종, 근거 간 평균 코사인 유사도 0.531

- [근거 2] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `8번 청크` · [Qwen3-8B-GGUF > Best Practices]
- [근거 3] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `49번 청크` · Chat Template Design. To better integrate the two modes and enable use
- [근거 4] (HF 모델 카드) Qwen/Qwen3-8B-GGUF 모델 카드 · `5번 청크` · [Qwen3-8B-GGUF > Switching Between Thinking and Non-Thinking Mode]
- [근거 5] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `51번 청크` · model’s thinking reaches a user-defined threshold, we manually halt th
- [근거 6] (arXiv PDF p.11) Qwen3 Technical Report (arXiv:2505.09388) · `46번 청크` · Table 9: Examples of SFT data for thinking and non-thinking modes duri
- [근거 7] (arXiv PDF p.12) Qwen3 Technical Report (arXiv:2505.09388) · `57번 청크` · 4.6 Post-training Evaluation
- [근거 8] (arXiv PDF p.21) Qwen3 Technical Report (arXiv:2505.09388) · `94번 청크` · questions are not factual and avoid generating hallucinatory answers.
- [근거 9] (arXiv PDF p.25) Qwen3 Technical Report (arXiv:2505.09388) · `112번 청크` · Qwen3-30B-A3B 75.9 94.5 81.9 80.0 48.1 76.1

## Qwen/Qwen3-4B — 긴 텍스트를 처리하려면 어떻게 설정해야 해?

**A. similarity (중복 제거 끔)** — 서로 다른 문서 3종, 근거 간 평균 코사인 유사도 0.480

- [근거 2] (위키백과) 컨텍스트 윈도 (ko.wikipedia) · `0번 청크` · 대형 언어 모델(LLM)의 컨텍스트 윈도(context window)는 모델이 출력을 생성할 때 한 번에 처리할 수 있는 텍스
- [근거 3] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `33번 청크` · === 음성 및 오디오 처리 ===
- [근거 4] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `18번 청크` · 로 쓸 수 있다. 이는 각 출력 토큰에 대한 교차 엔트로피 손실의 평균과 같다.
- [근거 5] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `20번 청크` · 를 계산한다. 디코딩 규칙(영어: decoding rule)에 따라 이 분포에서 하나의 토큰 [수식] isplaystyle {
- [근거 6] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `7번 청크` · === 입력 표현과 위치 정보 ===
- [근거 7] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `32번 청크` · === 자연어 처리 ===
- [근거 8] (HF 모델 카드) Qwen/Qwen3-4B 모델 카드 · `22번 청크` · [Streaming generation]
- [근거 9] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `21번 청크` · === 병렬 처리와 계산 복잡도 ===

**B. similarity + 근사 중복 제거** — 서로 다른 문서 3종, 근거 간 평균 코사인 유사도 0.480

- [근거 2] (위키백과) 컨텍스트 윈도 (ko.wikipedia) · `0번 청크` · 대형 언어 모델(LLM)의 컨텍스트 윈도(context window)는 모델이 출력을 생성할 때 한 번에 처리할 수 있는 텍스
- [근거 3] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `33번 청크` · === 음성 및 오디오 처리 ===
- [근거 4] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `18번 청크` · 로 쓸 수 있다. 이는 각 출력 토큰에 대한 교차 엔트로피 손실의 평균과 같다.
- [근거 5] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `20번 청크` · 를 계산한다. 디코딩 규칙(영어: decoding rule)에 따라 이 분포에서 하나의 토큰 [수식] isplaystyle {
- [근거 6] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `7번 청크` · === 입력 표현과 위치 정보 ===
- [근거 7] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `32번 청크` · === 자연어 처리 ===
- [근거 8] (HF 모델 카드) Qwen/Qwen3-4B 모델 카드 · `22번 청크` · [Streaming generation]
- [근거 9] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `21번 청크` · === 병렬 처리와 계산 복잡도 ===

**C. MMR + 근사 중복 제거 (기본)** — 서로 다른 문서 3종, 근거 간 평균 코사인 유사도 0.374

- [근거 2] (위키백과) 컨텍스트 윈도 (ko.wikipedia) · `0번 청크` · 대형 언어 모델(LLM)의 컨텍스트 윈도(context window)는 모델이 출력을 생성할 때 한 번에 처리할 수 있는 텍스
- [근거 3] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `33번 청크` · === 음성 및 오디오 처리 ===
- [근거 4] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `7번 청크` · === 입력 표현과 위치 정보 ===
- [근거 5] (HF 모델 카드) Qwen/Qwen3-4B 모델 카드 · `22번 청크` · [Streaming generation]
- [근거 6] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `21번 청크` · === 병렬 처리와 계산 복잡도 ===
- [근거 7] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `28번 청크` · 로 나타낼 수 있다. 이에 비해 Pre-LN(영어: pre-layer normalization) 방식은 [수식] erNorm 
- [근거 8] (위키백과) 트랜스포머 (기계 학습) (ko.wikipedia) · `26번 청크` · === 위치 표현 ===
- [근거 9] (HF 모델 카드) Qwen/Qwen3-4B 모델 카드 · `5번 청크` · [Qwen3-4B > Quickstart]
