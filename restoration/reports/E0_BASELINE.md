# E0 — 하네스 검증과 분할 정직성 실측

> 상태: `SEED_TRAINED_DIAGNOSTIC`
>
> 505개 dev 시드로 **학습까지** 한 일부러 잘못된 설정이다. 유효한 모델이 아니라,
> 분할 규칙의 근거와 코퍼스 스케일링 곡선의 최좌단 점을 만들기 위한 진단이다.

## 실행 조건

- 기법: chosung, tensify / 강도 (0.3, 0.5, 1.0)
- 모델: sklearn LogisticRegression(C=2.0, max_iter=400), 문자 창 +-4
- holdout source: `template_v1` (505 중 256)
- bootstrap 2,000회, 난수 시드 2026
- identity 페어(clean -> clean)를 학습에 포함

## 결과 — 변형이 실제로 걸린 위치만

| 기법 | 분할 | 초성 | 중성 | 종성 | 음절 exact | clean mutation |
|---|---|---|---|---|---|---|
| chosung | source_disjoint | 1.000 | 0.517 | 0.655 | 0.384 | 0.0% |
| chosung | random | 1.000 | 0.622 | 0.745 | 0.526 | 0.0% |
| tensify | source_disjoint | 0.989 | 1.000 | 1.000 | 0.989 | 23.8% |
| tensify | random | 0.971 | 1.000 | 1.000 | 0.971 | 14.8% |

## 다수결 baseline 대비 (source 분할, changed-only)

| 기법 | 슬롯 | 모델 | 다수결 |
|---|---|---|---|
| chosung | jung | 0.517 | 0.115 |
| chosung | jong | 0.655 | 0.563 |
| tensify | cho | 0.989 | 0.331 |

## 랜덤 분할이 부풀리는 양 (랜덤 − source, %p)

| 기법 | 초성 | 중성 | 종성 | 음절 exact |
|---|---|---|---|---|
| chosung | +0.0 | +10.6 | +8.9 | +14.2 |
| tensify | -1.8 | +0.0 | +0.0 | -1.8 |

## 판정

- 이후 모든 실험은 **source-disjoint 분할**만 주 지표로 보고한다. 랜덤 분할 수치는
  대조군으로만 병기한다.
- clean mutation rate가 승격 기준(<=1%)에서 얼마나 먼지가 E1의 출발점이다.

## 해석 제한

- 505 시드로 학습한 설정이라 어휘 폭이 극단적으로 좁다. 절대 수치를 모델 성능으로
  읽으면 안 된다.
- 가드레일 추론을 하지 않았으므로 NRR·ΔFPR은 이 문서에 없다.
- `results/`의 `predictions.jsonl`에는 공격 원문이 들어 있어 커밋하지 않는다.
