# 복원기 실험 트랙

난독화된 한국어(된소리, 연음, 종성 크래밍 등)를 원문 후보로 되돌리는 복원기를 실험한다.
설계 견적 원본은 [`PLAN.md`](./PLAN.md)(구 `k-safeguard_ml_restoration_plan.md`).

- **주 목적**: k-safeguard에 탑재될 복원기의 성능 최대화.
- **부 목적**: 복원기가 다루는 난독화 종류(커버리지) 확대.

## 먼저 읽을 것

1. [`reports/SUMMARY.md`](./reports/SUMMARY.md) — E0~E10 전체를 묶은 종합 판정과 저장소에
   돌려줄 산출물.
2. [`EXPERIMENTS.md`](./EXPERIMENTS.md) — 실험별 한 줄 대장. 개별 상세는 [`reports/`](./reports/).
3. [`notes/`](./notes/) — 2026-10 이후 단계별 실험 이유·해석·리뷰 기록.

## 이 트랙의 규칙

- **505개 dev 시드는 평가 전용.** 학습에 넣지 않는다(E0만 진단 목적의 예외).
- **복원기는 후보만 낸다.** block/allow 판단은 하지 않는다.
- **가드레일 실측은 가드레일 트랙과 나눠 본다.** E10처럼 복원기를 실제 가드레일에 물려 재는
  실험은 이 트랙에 두고(복원기가 대상), 가드레일 자체를 고르거나 비교하는 실험은
  [`guardrail/`](../guardrail/README.md)에 둔다. E10의 Colab 노트북은
  [`notebooks/`](./notebooks/)에 있다.

## 실행

```bash
source ./env.sh                                   # 저장소 루트에서
python -m restoration.exp.stage status
python -m restoration.exp.run_e8_abstention --run-id <run-id> --corpus-size 30000
```

## 지표와 승격 기준

승격 후보가 되려면 셋을 모두 넘어야 한다.

1. **Clean Mutation Rate ≤ 1%** — 난독화되지 않은 정상 문장을 복원기에 통과시켰을 때
   바뀌는 문장 비율. `EVALUATION_SPEC.md` §11.4 기준 그대로다. **1차 게이트**이고,
   이걸 못 넘으면 복원 정확도가 아무리 좋아도 승격하지 않는다 — 된소리 provider가
   정확히 이 축(ΔFPR-obf +14.29%p)에서 `DO_NOT_PROMOTE`가 됐다.
2. 해당 기법의 음절 exact가 현행 규칙/사전 방식 대비 유의하게 높을 것.
3. CPU p95 지연 ≤ 100ms(MLP) / ≤ 200ms(BiLSTM).

## 지금 상태 (2026-10-14)

- s01(E8b, BiLSTM + abstention)은 승인으로 닫혔다.
- 다음 단계는 아직 만들지 않았다. 논의 중인 방향은 "복원기가 자체 난독화 생성기에 얼마나
  묶여 있는가"를 먼저 재는 것이고, 외부 기준으로 KOTOX를 쓰기로 했다([`DATASETS.md`](../DATASETS.md)).
