# 실험 인덱스 — 가드레일 트랙

어떤 실험이 있었고 각 실험의 코드·기준값·보고서·원자료가 어디 있는지, 이름끼리 어떻게 대응하는지를 적는다.
수치와 결론은 [`EXPERIMENTS.md`](./EXPERIMENTS.md)(실험 대장)와 `reports/`가 원본이다. 이름 짓는 규칙은
[`NAMING.md`](./NAMING.md). 가드레일 후보 조사는 [`CANDIDATES.md`](./CANDIDATES.md).

## 한눈에 보기

| 단계 | 로드맵 작업 | 질문 | run | 상태 | 보고서 |
|---|---|---|---|---|---|
| **S01 난독화 강건성** (`s01-obf-robustness`) | P02 가드레일 다중 | 어떤 난독화 유형이 인젝션 가드레일을 통과하나 | 6 (결과 4, 중단 2) | 리뷰 전 | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
| **S02 LLM 이해도** (`s02-llm-understanding`) | P03 LLM 강건성 | 난독화된 공격을 LLM이 알아듣나 | 2 + 판정자 | 리뷰 전 | [S02](./reports/S02_LLM_UNDERSTANDING.md) |
| **P05 욕설 필터 난독화 내성과 복원** (`p05-profanity-filter`) | P05 욕설 필터 | 공개 욕설 필터가 얼마나 뚫리고 복원기가 얼마나 되찾나 | 4 | 승인됨 | [P05](./reports/P05_PROFANITY_FILTER.md) |
| **P05 섞은 난독화** (`p05-mixed-obfuscation`) | P05 욕설 필터 | 기법을 섞으면 더 뚫리나, 복원기·OR 결합이 되찾나 | 8 | 승인됨 | [P05 5절](./reports/P05_PROFANITY_FILTER.md) |
| **P02 최종 복원기 + Kanana** (`p02-kanana-restorer`) | P02 가드레일 다중 | 최종 복원기를 붙이면 S01 Kanana 우회율이 얼마나 주나 | 1 | 리뷰 전 | [P02](./reports/P02_KANANA_RESTORER.md) |
| **P03 Tensor Trust 비밀값 추출** (`p03-secret-extraction`) | P03 LLM 강건성 | 실제 추출 공격이 한국어 LLM과 가드레일에 통하나(원문·번역본) | 10 (`p03.3`~`p03.12`) | 승인됨 | [P03](./reports/P03_SECRET_EXTRACTION.md) |
| **P03 번역본 난독화 비밀값 추출** (`p03-obf-extraction`) | P03 LLM 강건성 | 번역본 공격을 난독화하면 가드레일 앞단 유출이 오르나 | 6 (`p03.13`~`p03.18`) | 승인됨 | [P03 난독화](./reports/P03_OBF_EXTRACTION.md) |
| **P03 모델 크기·난독화 기법 확장** (`p03-model-scale`) | P03 LLM 강건성 | 큰 모델·더 많은 기법·적응형 공격에서 앞단 유출이 오르나 | 52 (결과 49, 미실행 3) | 승인됨 | [P03 확장](./reports/P03_MODEL_SCALE.md) |

두 단계 모두 로드맵의 작업 번호(P 시리즈) 대신 `s` 접두어로 열렸다. id는 바꾸지 않고 위 대응만 둔다.
다음 단계부터는 `p02-…`, `p03-…`를 쓴다.

## S01 난독화 강건성

평가셋 v1 전체(난독화 24,444 + 원문 695)를 가드레일 하나씩에 태워 유형별 회피율과 정상 오차단율을 쟀다.

| run id | 대장 표기 | 가드레일 | 결과 | 기준값 |
|---|---|---|---|---|
| `s01.1-kanana` | S01.1 | Kanana safeguard prompt 2.1b | 있음 | [`s01_kanana_v1.json`](./baselines/s01_kanana_v1.json) |
| `s01.2-horizon-base` | S01.2 | Horizon-Labs injection guard base | 있음 | [`s01_horizon_base_v1.json`](./baselines/s01_horizon_base_v1.json) |
| `s01.3-horizon-large` | S01.3 | Horizon-Labs injection guard large | 있음 | [`s01_horizon_large_v1.json`](./baselines/s01_horizon_large_v1.json) |
| `s01.4-klue-roberta` | S01.4 | KLUE RoBERTa 인젝션 분류기 | 있음(비교 제외) | [`s01_klue_roberta_v1.json`](./baselines/s01_klue_roberta_v1.json) |
| `s01.5-gemma4-e2b` | S01.5 | Gemma 4 E2B, 생성 판정 | 중단 | — |
| `s01.5b-gemma4-e2b-logit` | S01.6 | Gemma 4 E2B, 확률 판정 | 중단 | — |

- 러너: `exp/run_s01_obf_robustness.py`. 가드레일 정의는 `scorers.py`, 평가셋 읽기·집계는 `evalset.py`.
- 원자료: run별 `results/<run-id>/`(문장별 점수 `scores.jsonl` 포함). 실행한 머신에만 있고 저장소에는 없다.
- 이어 쓰는 곳: 문장별 점수는 S02의 가드레일 앞단 계산과 복원기 트랙 P04가 다시 읽는다.

## S02 LLM 이해도

평가셋 v1의 고정 부분집합(2,310)과 원문(695)을 한국어 LLM에 넣고, 응답을 LLM 판정자로 채점했다.
S01 점수와 문장 단위로 합쳐 "가드레일 통과 + LLM 이해" 위협률도 냈다.

| run id | 대장 표기 | 보고서 표기 | 역할 | 모델 |
|---|---|---|---|---|
| `s02.1-exaone4-32b-awq` | S02.1 | P03.1 | 대상 | EXAONE 4.0 32B AWQ |
| `s02.2-kanana15-8b` | S02.2 | P03.2 | 대상 | Kanana 1.5 8B instruct |
| (id 없음) | — | — | 판정자 | Qwen3 32B AWQ |

- 스크립트: `p03/`(생성 → 판정 → 집계 → 검수, 가드레일 앞단 결합). 각 파일의 역할은 docstring에 있다.
- 기준값: [`s02_llm_v1.json`](./baselines/s02_llm_v1.json) 하나에 두 모델과 가드레일 결합 결과가 함께 들어 있다.
- 원자료: Colab에서 실행했고 비공개 Drive에 있다. 공격 문장에 대한 응답이 들어 있어 공개하지 않는다.
- 단계 관리 도구 밖에서 실행했다. 그래서 대장의 단계 표에 S02 행이 없다.
- 부분집합 파일(`p03_subset.jsonl`)은 Colab 작업 폴더에 있고 저장소에는 선택 규칙만 보고서에 있다.

## 이름이 어긋난 곳

id는 바꾸지 않는다. 앞으로 새로 짓는 이름에서 반복하지 않도록 기록만 해 둔다.

| 어디 | 어긋남 | 읽는 법 |
|---|---|---|
| 단계 접두어 | 로드맵은 2기 단계를 `p02-…`·`p03-…`로 정했는데 S01·S02는 `s`로 열렸다. 복원기 트랙의 S01~S03과도 겹친다 | 가드레일 S01 = P02의 첫 단계, 가드레일 S02 = P03의 첫 단계 |
| S01 Gemma 확률 판정 | id는 `s01.5b`, 대장 표기는 S01.6 | 같은 run이다. 대장 번호는 등록 순서에서 나왔다 |
| S02 run 번호 | 대장은 S02.1·S02.2, 보고서는 P03.1·P03.2 | 같은 run이다 |
| S02 코드 폴더 | 단계는 S02인데 코드는 `p03/`에 있고 러너가 `exp/` 밖에 있다 | 작업 번호(P03)로 묶은 것 |
| S02 결과 경로 | 스크립트 예시는 run id 폴더가 아니라 `results/s02-llm/` 한 폴더에 두 모델을 둔다 | 파일 이름의 모델 줄임말로 run을 가른다 |
| 대장 단계 표 | S01 상태가 "실행 중"으로 남아 있다(run은 모두 끝남) | 단계 표는 도구가 다시 쓴다. 리뷰가 끝나면 갱신된다 |
