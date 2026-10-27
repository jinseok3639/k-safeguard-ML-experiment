# S01 — 난독화 강건성: 인젝션 가드레일 4종

> 상태: `PROVISIONAL_DEV_ONLY` (리뷰 전. Gemma 4 E2B 미측정)
>
> kslab 평가셋 v1(난독화 24,444문장 + 원문 695문장)을 인젝션 가드레일 4종으로 채점했다.
> 결론: **Kanana는 원문 공격을 가장 잘 막지만(94.0%) 난독화에 가장 많이 뚫린다**(원문은
> 막혔던 공격의 27.5%가 난독화본에서 통과). 다국어 인코더(Horizon)는 덜 뚫리지만(14~15%)
> **난독화된 정상 문장을 원문보다 2.4~6.5배 많이 차단한다.** 두 종류 모두에서 받침·반모음
> 추가(`phon_insert`)가 상위 위협이다.

## 실행 환경

- 로컬 RTX 5070 Ti 16GB, Python 3.10.0, torch 2.9.1+cu128, transformers 4.57.6, fp16
- 커밋 `516151c`(`experiment/s01-obf-robustness`). run별 manifest·pip-freeze는 `results/<run-id>/`
- 평가셋: kslab-evalset-v1. `records.jsonl` sha256 `3da43bbf…`. 원문 695개(공격 301, 정상 394)
- 기준값 0.5. 원문 단위 bootstrap 2,000회
- 집계: `types.csv` 16유형 + 대조군. 한 문장에 유형이 여럿이면 유형마다 한 번씩 센다
- 아래 백분율은 원문 균형 추정치다. 원문마다 비율을 먼저 구하고 원문끼리 평균했다. 행 단위 비율과 95% CI는 `baselines/s01_*_v1.json`에 있다

| run | 가드레일 | 리비전 | 소요 | GPU 메모리 |
|---|---|---|---|---|
| S01.1 | `kakaocorp/kanana-safeguard-prompt-2.1b` | `167d74d` | 299초 | 11.3GB |
| S01.2 | `Horizon-Labs/prompt-injection-guard-base` | `70b65c5` | 49초 | 0.8GB |
| S01.3 | `Horizon-Labs/prompt-injection-guard-large` | `8ae7c38` | 115초 | 1.3GB |
| S01.4 | `kor-prompt-injection-detection/KLUE_roBERTa_based_model` | `788922d` | 30초 | 0.4GB |

## 1. 요약

| 가드레일 | 원문 탐지율 | 원문 오차단율 | 난독화 회피율 | 회피\|원문차단 [95% CI] | 난독화 오차단율 |
|---|---|---|---|---|---|
| Kanana | **94.0%** | 3.8% | 30.8% | **27.5** [25.6, 29.6] | 2.8% |
| Horizon base | 90.7% | 1.3% | 18.5% | 13.8 [12.3, 15.4] | **8.4%** |
| Horizon large | 92.4% | 2.8% | 18.8% | 14.6 [12.9, 16.3] | 6.8% |
| KLUE RoBERTa (참고) | 74.1% | 26.6% | 53.8% | 42.3 [40.0, 44.6] | 26.0% |

- **회피\|원문차단**은 원문이 차단된 공격만 놓고 본 회피율이다. 원문부터 못 막는 공격은 난독화의 효과가 아니므로 이 값을 주 지표로 쓴다.
- KLUE RoBERTa는 원문 오차단율이 26.6%로 사전 기준(20%)을 초과한다. 비교 대상에서 빼고 참고로만 싣는다.
- Horizon base와 large는 회피\|원문차단 CI가 겹친다. 모델 크기를 키워도 난독화 내성은 늘지 않았다.

## 2. 유형별 회피\|원문차단(%)

| 유형 | 분류 | 복원기 범위 | Kanana | Horizon base | Horizon large | KLUE(참고) |
|---|---|---|---|---|---|---|
| phon_initial | 초성 교체·된소리 | target | **41.0** | 17.3 | 17.4 | 34.0 |
| phon_vowel | 모음 교체 | target | 28.7 | 16.8 | 17.8 | 41.7 |
| phon_coda | 받침 교체 | target | 25.4 | 14.5 | 15.1 | 28.1 |
| phon_spelling | 소리 나는 대로 | target | 28.6 | 14.6 | 17.7 | 31.5 |
| phon_insert | 받침·반모음·초성 추가 | target | **41.7** | **24.0** | **24.0** | 45.5 |
| phon_liaison | 연음 | target | 29.1 | 16.9 | 18.6 | 43.8 |
| shape_foreign | 닮은 비한글 문자 | boundary | **49.2** | 5.2 | 15.8 | 67.0 |
| shape_hangul | 야민정음 | boundary | **49.0** | 16.2 | 20.3 | 33.6 |
| translit | 라틴·한자 음차 | boundary | **44.1** | 20.4 | 25.6 | 36.6 |
| spacing | 띄어쓰기 교란 | target | 21.8 | 12.8 | 15.0 | 49.4 |
| anagram | 글자 순서 섞기 | boundary | **48.0** | **25.2** | **28.9** | 36.7 |
| symbol_insert | 기호 삽입 | target | 23.4 | 9.5 | 10.6 | 52.3 |
| jamo_decompose | 자모 분해 | target | 16.4 | 7.8 | 11.9 | 74.7 |
| syllable_extend | 음절 늘이기 | boundary | 25.2 | 12.0 | 13.0 | 44.0 |
| chosung | 초성체 | **out** | 50.0 | 17.3 | 19.1 | 91.5 |
| control | 키보드 오타(대조군) | - | 5.0 | 5.1 | 3.6 | 16.1 |

- 대조군(무작위 키보드 오타)은 세 가드레일 모두 5% 안팎이다. 규칙을 따르는 난독화는 모든 유형이 대조군보다 높다.
- Kanana에서는 모양 계열(`shape_*`, `translit`, `anagram`)이 44~49%로 가장 높고, 음운 계열 중에서는 `phon_initial`·`phon_insert`가 41%다.
- Horizon에서는 `phon_insert`·`anagram`이 24~29%로 가장 높다. `shape_foreign`은 회피가 낮은 대신 오차단이 높다(3절).
- **`chosung`은 복원기 범위 밖이다**(2026-10-27 초성 복원 제외 결정). 가드레일 측정에는 넣었지만 복원기 우선순위 판단에서는 뺀다.

### 단일 기법 상위(회피\|원문차단, single 축)

| 기법 | 유형 | Kanana | Horizon large |
|---|---|---|---|
| `jongseong_cram` | phon_insert | **76.2** | **48.4** |
| `semivowel_insert` | phon_insert | 55.7 | 32.5 |
| `lookalike_foreign` | shape_foreign | 48.6 | 13.4 |
| `tensify` | phon_initial | 47.9 | 18.4 |
| `jamo_copy` | phon_insert | 13.8 | 14.1 |

`phon_insert` 유형의 높은 회피율은 대부분 `jongseong_cram`(받침 채우기)과 `semivowel_insert`에서 나온다. 같은 유형의 `initial_insert`는 Kanana 8.0%, Horizon large 2.6%다. 발동 수가 50개 미만인 기법(`palatalize` 6, `lateralize` 14, `h_deletion` 23, `nasalize` 41)은 표에서 뺐다.

## 3. 난독화된 정상 문장의 오차단율(%)

| 가드레일 | 원문 | 난독화 전체 | 가장 높은 유형 |
|---|---|---|---|
| Kanana | 3.8 | 2.8 | phon_liaison 4.1 |
| Horizon base | 1.3 | 8.4 | **shape_foreign 24.1**, translit 14.2 |
| Horizon large | 2.8 | 6.8 | shape_foreign 11.5, symbol_insert 11.5 |

Kanana는 난독화로 오차단이 늘지 않는다. Horizon은 정상 문장을 난독화하면 오차단이 원문의 2.4~6.5배가 된다. 낯선 문자열을 인젝션으로 보는 경향으로 해석된다. 복원기를 Horizon 앞에 붙이면 회피 회복과 함께 오차단 감소도 기대할 수 있으므로 P04에서 두 효과를 함께 잰다.

## 4. 기법 수(compose·external)

| 실제 걸린 기법 수 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Kanana | 35.3 | 19.9 | 33.3 | 47.5 |
| Horizon large | 14.0 | 12.5 | 18.8 | 23.5 |

2개 이상에서는 기법 수가 늘수록 회피가 오른다. 1개 칸은 external 축의 kotka 단일 변환이 대부분이라 single 축과 성격이 다르다. 그래서 1→2 감소는 해석하지 않는다.

## 5. 복원기 우선순위 제안

사전 기준: Kanana 회피\|원문차단 30% 이상이면 복원기 우선 후보, 10% 미만이면 우선순위에서 내린다.

| 판정 | 유형 |
|---|---|
| 우선 후보(복원기 범위 target) | `phon_insert`(41.7), `phon_initial`(41.0) |
| 우선 후보(범위 boundary, 범위 결정 필요) | `shape_foreign`(49.2), `shape_hangul`(49.0), `anagram`(48.0), `translit`(44.1) |
| 제외(범위 out) | `chosung`(50.0). 초성 복원 제외 결정 |
| 10% 미만 | 없음 |

target 유형 중에서는 `phon_insert`가 Kanana와 Horizon 양쪽에서 가장 높다. 그 안에서도 `jongseong_cram`·`semivowel_insert` 두 기법이 우선이다.

## 해석 제한

- **평가셋은 생성기로 만든 난독화다.** 사람이 실제로 쓴 난독화가 아니다. 유형별 순위는 생성 강도(single 0.5, compose 0.3)에 따라 바뀔 수 있다. restoration E10에서 된소리 단독 회피는 85.5%였고 여기 `tensify`는 47.9%다. 강도와 시드 구성이 달라 직접 비교하지 않는다.
- **유형마다 문장 수와 원문 구성이 다르다.** `translit`·`anagram`은 external 축(KOTOX·kotka)에만 있다. 다른 유형과 같은 줄에 놓고 비교할 때 생성기 차이가 섞인다.
- **공격 원문은 301개다.** 회피\|원문차단은 원문이 차단된 공격(Kanana 283개, Horizon base 273개, large 278개)만 쓴다. 가드레일마다 분모가 다르다.
- **기준값은 0.5 하나다.** 기준값을 바꾸면 회피율과 오차단율이 함께 움직인다. 이 결과로 가드레일 간 우열을 정하지 않는다.
- **fp16 수치 차이.** Kanana 원문 오차단율은 fp32(CPU) 3.8%, bf16(GPU 스모크) 4.1%, fp16(이 run) 3.8%다. 경계 근처 1~2문장이 정밀도에 따라 바뀐다.
- **Gemma 4 E2B는 측정하지 않았다.** 로컬 16GB GPU에서 생성 방식은 1시간 이상 걸려 중단했다. 순전파 1회 확률 판정 방식(생성 방식과 판정 일치 398/400)은 VRAM 초과와 속도 저하로 두 번 중단했다. 결과 수치는 보지 않았다. "가드레일이 난독화에 강하면 복원기가 필요 없다"는 반론에는 이 보고서로 답하지 않는다.
- 문장별 점수(`results/<run-id>/scores.jsonl`)는 공개하지 않는다. 평가셋 id로 P04에 전달한다.
