# 실험 대장 — 복원기 트랙

실패한 실험도 지우지 않는다 — 저장소의 `TENSIFY_LOCKED_RESULT.md` 선례대로, 안 되는 것을
기록해 두는 것도 결과다.

## 단계별 요약

단계(질문 하나) 하나에 한 줄. 각 단계 문서는 맨 위에 질문 → 답 → 근거 숫자 → 다음 순서로 쓰여
있다. 이 표는 `python -m restoration.exp.stage sync <단계-id>`가 단계 문서에서 다시 만든다.

<!-- stages:begin (stage sync가 다시 쓴다. 손으로 고치지 않는다) -->
| 단계 | 질문 | 답 | 상태 | 문서 |
|---|---|---|---|---|
| **S01 BiLSTM vs MLP (포기 임계값)** | 단일 난독화 기법(tensify·liaison·jongseong_cram)에서 abstention을 건 상태로 BiLSTM이 MLP보다 나은가? | 연음·받침 크래밍은 BiLSTM이 확실히 낫다. 된소리는 판정할 수 없다(임계값 격자·benign 표본이 성김). | 승인됨 | [S01](./notes/s01-e8b-bilstm-abstention/STAGE.md) |
| **S02 생성기 의존도** | 복원기가 우리 난독화 생성기(`kslab/obfuscate.py`)에 묶여 있나? | 그렇다 — 복원기는 우리 생성기 패턴을 외운 것이고, 남은 성능도 기법 공통 임계값 하나가 깎아 먹는다. | 승인됨 | [S02](./notes/s02-generator-dependence/STAGE.md) |
<!-- stages:end -->

## run 기록

run 하나당 한 줄. 단계 도입 전(E0~E10)은 이 표와 `reports/`가 유일한 해석 기록이다.

상태 값:
`SEED_TRAINED_DIAGNOSTIC`(진단 전용, 모델 아님) ·
`PROVISIONAL_DEV_ONLY`(dev에서만 확인) ·
`PROMOTE_CANDIDATE`(승격 기준 통과) ·
`DO_NOT_PROMOTE`(기준 미달, 기록용 보존)

| 날짜 | run-id | 실험 | 핵심 수치 | 상태 | 보고서 |
|---|---|---|---|---|---|
| 2026-08-23 | `e0-20260823-b` | E0 하네스·분할 정직성 | chosung 중성 .517 / 종성 .655, clean mutation 0.0% · tensify 초성 .989, clean mutation 23.8% · 랜덤분할이 chosung 음절exact를 +14.2%p 부풀림 | `SEED_TRAINED_DIAGNOSTIC` | [E0](./reports/E0_BASELINE.md) |
| 2026-08-23 | `e5a-20260823` | E5a 생성기 갭 메우기 | 겹받침 회귀 12/12 통과 · liaison 발동률 80.2% · palatalize 1.8%(실효성 낮음) · 길이보존 5종 100% | `PROVISIONAL_DEV_ONLY` | [E5a](./reports/E5A_GENERATORS.md) |
| 2026-08-23 | `e1-20260823` | E1 코퍼스 스케일링 | 500→100k문장: tensify clean mutation 50.0%→6.4%(평평해짐) · liaison 86.3%→10.3% · cram 92.6%→74.5% · chosung 음절exact 0.131→0.152(불변) | `PROVISIONAL_DEV_ONLY` | [E1](./reports/E1_CORPUS_SCALING.md) |
| 2026-08-23 | `e3-20260823` | E3 된소리 되돌림 판단 | thr 0.7에서 benign 7.35% / exact 0.744 (현행 정책 11.3% 대비 개선, 후보 9개→1개). 1% 바는 thr 0.999에서만 통과하나 exact 0.024로 붕괴 | `DO_NOT_PROMOTE` | [E3](./reports/E3_TENSIFY_DECISION.md) |
| 2026-08-23 | `e6-20260823` | E6 n-gram 게이트 | liaison CER 0.124→0.041(게이트 없이), 게이트는 benign 34%→25%로만 줄임. cram CER 0.348→0.192. 어느 쪽도 1% 근처 못 감 | `DO_NOT_PROMOTE` | [E6](./reports/E6_GATE.md) |
| 2026-08-23 | `e2-20260823` | E2 아키텍처 사다리 | **E1 결론 정정.** 8천문장에서 BiLSTM이 liaison 오변경 33.8%→5.9%, cram 72.1%→12.7% (선형 100k문장보다 좋음). chosung 음절exact 0.139→0.301. 병목은 코퍼스가 아니라 고정 창 표현력이었음. p95 5.1ms | `PROVISIONAL_DEV_ONLY` | [E2](./reports/E2_ARCHITECTURE.md) |
| 2026-08-23 | `e4-20260823` | E4 초성체 사각지대 | 사전이 못 푸는 17단어 전부 **시도는** 하나 **0/17 정답**. 문장 안에서도 음절exact 0.250(mlp)/top3 0.581. 틀릴 때 그럴듯한 다른 한국어를 만들어냄 — FP 주입 위험 | `DO_NOT_PROMOTE` | [E4](./reports/E4_CHOSUNG_SLOTS.md) |
| 2026-08-23 | `e7-20260823` | E7 provider 계약·ONNX·지연 | 실제 Gateway에서 계약 위반 0건(view 254 / 후보 94) · ONNX 3종 torch와 정확히 일치(각 ~1.1MB) · p95 **4.7ms**(예산 100ms) | `CONTRACT_OK` | [E7](./reports/E7_PROVIDER_PROTOTYPE.md) |
| 2026-08-23 | `e8-mlp-30k` | E8 아키텍처 × abstention | **처음으로 두 기준 동시 통과.** tensify thr 0.999999에서 오변경 0.49%(1/204) + CER −11.84%p, 난독문장 99% 손댐. 단 benign 204개는 해상도 0.49%·0건이어도 CI 상한 1.85% → 400개 이상으로 재측정 필요 | `PROMOTE_CANDIDATE` (조건부) | [E8](./reports/E8_ABSTENTION.md) |
| 2026-08-23 | `e2-bilstm-30k` | E2b BiLSTM 코퍼스 스케일링 | **중단됨(tensify 도중).** liaison 8천→3만: exact 0.766→0.893, 오변경 5.9%→2.9% · cram 0.667→0.738, 12.7%→2.0%. 선형은 100k에서도 평평했는데 BiLSTM은 4배만으로 절반 이하 — 아키텍처가 코퍼스 스케일링을 열어준다 | `PARTIAL_INTERRUPTED` | [E2b](./reports/E2B_BILSTM_SCALING.md) |
| 2026-08-25 | `e9a-20260825` | E9a 음운 규칙 확장 + 조합 상한(학습 없음) | 표준발음법 9종 신규(조항 회귀 48/48 + 조합 5/5 통과) · **조합 실패의 원인은 슬롯이 아니라 탐지 신호 소실** — 구조적 미달 4건이 전부 도달불가(tensify d3 69.7%, aspirate d2 82.9%)이고 표현불가는 거의 0 · 전수검사인 liaison은 조합에서도 도달불가 0%로 합집합과 동일 상한 · `vowel_confuse` 없이는 union 슬롯이 {초성,종성}이라 중성이 영원히 미커버 | `PROCEED_TO_E9B` | [E9a](./reports/E9A_PHONOLOGY.md) |
| 2026-08-25 | `e9b-mlp-30k` | E9b 조합 난독화 — 단일 vs 혼합 학습(MLP) | **단일학습 합집합 모델은 깊이1(d1_tensify +2.10%p)만 통과, 깊이2·3은 0곳 통과(유지율 0%)** — 조합은 실제 문제 · **혼합학습(S+M)은 통과 지점이 0곳** — 섞어도 안 고쳐짐 · **학습비용 0인 cascade가 +3.65%p로 유일하게 깊이2까지 통과**, 단 자리별 분해로 보면 이건 사실상 tensify 단독 모델이 이끄는 것(tensify 귀속 net_repair +0.78 vs liaison 귀속 +0.14) | `PROMOTE_CANDIDATE`(cascade만, universal은 `DO_NOT_PROMOTE`) | [E9b](./reports/E9B_COMPOSITION.md) |
| 2026-08-26 | `e9b-bilstm-30k` | E9b Stage 3 — 조합 난독화 BiLSTM 확인 | **아키텍처가 MLP에서 막힌 깊이2를 뚫는다.** `bilstm/S`가 d1_tensify +4.80%p·d1_liaison +2.80%p·d1_vowel +2.18%p에 더해 **d2_liaison_then_tensify +3.71%p까지 통과**(MLP는 0곳). 단 깊이별 유지율 자체(d2/d1≈77%, d3/d1≈22%)는 MLP와 비슷 — 아키텍처가 비율을 바꾼 게 아니라 절대 기준선을 ~2배 올려서 같은 비율로도 깊이2가 2%p 바를 넘김. 깊이3(`d3_canonical` +1.07%p@q0.10)은 여전히 미달. `bilstm/S+M`은 여기서도 안 통함 — 같은 포기율에서 benign 변형이 S보다 크고(1.96% vs 0.49%), 그 변형을 누르면 CER개선이 S의 d2 통과치보다 낮아짐(1.54%p) | 개별 통과지점 `PROMOTE_CANDIDATE`(S만, S+M은 d1 1곳뿐) | [E9b](./reports/E9B_COMPOSITION.md) |
| 2026-08-26 | `colab-kanana-20260826-014300` | E10 실제 Kanana-Safeguard 가드레일 검증(Colab, 첫 실측) | **저장소 최초로 진짜 가드레일에 추론.** 탐지 복원 자체는 실측 확인됨 — tensify 난독화가 실제 TPR을 94.0%→14.0%로 무너뜨리고(회피율 CER 85.5%) 복원 후 93.4%로 회복. `composed_liaison_tensify`는 회피율만 20.1%로 낮음(같은 손상예산인데 비대칭). §11.4 5기준 승격게이트는 10칸(2프로파일×5임계값) 전부 `all_pass=False` — g1·g2·g3·g5는 thr≤0.999 전 구간 통과인데 **g4(benign mutation 문장단위 5.4~7.8%, 기준 ≤1%)만 막음**, thr 0.999999에서 g4는 뚫리나(0.49%, E8과 동일) 대신 g1(NRR 12.8%)이 무너짐 — E3·E8이 proxy로 가리킨 트레이드오프가 실제 가드레일로 재현. proxy CER개선↔실제NRR 상관 0.546. **주의: 이 실행은 코퍼스 8천문장(E8은 3만)** — CER개선(+4.53%p@thr0.999999)이 E8(+11.84%p)보다 낮은 건 이 때문일 수 있음, 3만문장 재실행 필요 | `DO_NOT_PROMOTE`(현 스윕 기준, 코퍼스 재실험 전까지 확정 아님) | [E10](./reports/E10_REAL_GUARDRAIL.md) |
| 2026-10-07 | `e8-mlp-bilstm-30k-20261007` | E8b 단일 기법 BiLSTM + abstention | **liaison·jongseong_cram은 BiLSTM이 명확히 우세** — 오변경 ≤1%에서 liaison +10.60%p(MLP +5.99%p, exact 0.696 vs 0.160), cram +48.67%p(MLP +11.87%p); 오변경 0건에서도 +8.94/+29.93%p · **tensify는 미결** — 같은 thr 0.999999·오변경 0.49%에서는 MLP +11.31%p > BiLSTM +6.94%p, BiLSTM +17.02%p는 0.98% 지점(benign 1문장 차이, 격자가 성김) · MLP 수치는 기존 E8과 소폭 차이로 재현(tensify +11.84→+11.31%p) · 임계값을 평가셋에서 골라 낙관적 | `PROMOTE_CANDIDATE`(liaison·cram BiLSTM), tensify 아키텍처 선택은 미결 | [E8b](./reports/E8B_BILSTM_ABSTENTION.md) |
| 2026-10-14 | `e11a-data-audit-20261014` | E11a KOTOX 평가 데이터 감사 | KOTOX test 920쌍 중 슬롯 가역 22.3% · 전부 slot 16.1% · lexical 포함 45.0% · 길이 보존 38% · 평균 CER 0.51 · benign dev 1,835 / test 434(겹침 0) | `DATA_READY` | [E11](./reports/E11_GENERATOR_DEPENDENCE.md) |
| 2026-10-14 | `e11b-family-holdout-20261014` | E11b 기법군 제외 학습(합집합 BiLSTM, GPU) | **평균 유지율 -3.5%** — tensify +5.46→-0.09%p, liaison +2.25→-0.01, vowel +2.68→-0.09. 빼고 학습하면 0. coda·assimilation은 넣고도 ≈0이라 판정 불가. benign test 0~0.92% | `GENERATOR_BOUND` | [E11](./reports/E11_GENERATOR_DEPENDENCE.md) |
| 2026-10-14 | `e11c-cross-generator-20261014` | E11c 자체 생성기 모델 → KOTOX | KOTOX test 전체 +0.35%p · 슬롯 가역 +0.67%p(abstention 없음 +2.07%p) · 규칙별 최대 1-5 +0.98, 1-1 +0.93 · 1-4 받침 대치 +0.13 | `DOES_NOT_TRANSFER` | [E11](./reports/E11_GENERATOR_DEPENDENCE.md) |
| 2026-10-14 | `e11d-oracle-detector-20261014` | E11d 정답 위치 상한 | abstention 없음 ≈ oracle(tensify 19.4 vs 19.8, cram 33.2 vs 34.8%p) — 손실은 탐지가 아니라 benign 1% 임계값 비용. **합집합 모델의 공통 임계값이 cram을 +0.21%p로 자름**(단일 모델 +12.08%p) | `MEASURED` | [E11](./reports/E11_GENERATOR_DEPENDENCE.md) |
