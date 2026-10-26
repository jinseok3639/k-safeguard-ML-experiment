# 복원기 실험 트랙

난독화된 한국어(된소리, 연음, 종성 크래밍 등)를 원문 후보로 되돌리는 복원기를 실험한다.

- **주 목적**: k-safeguard에 탑재될 복원기의 성능 최대화.
- **부 목적**: 복원기가 다루는 난독화 종류(커버리지) 확대.

## 먼저 읽을 것

1. [`reports/SUMMARY.md`](./reports/SUMMARY.md) — E0~E10 전체를 묶은 종합 판정과 저장소에
   돌려줄 산출물.
2. [`EXPERIMENTS.md`](./EXPERIMENTS.md) — 단계별 질문·답과 run별 핵심 수치. 개별 상세는 [`reports/`](./reports/).

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

## 지금 상태 (2026-10-26)

- S01~S03이 닫혔다. 요약은 [`EXPERIMENTS.md`](./EXPERIMENTS.md) 단계 표.
- S02에서 복원기가 자체 난독화 생성기 패턴에 묶여 있음을 확인했고(외부 기준: KOTOX,
  [`DATASETS.md`](../DATASETS.md)), S03에서 임계값 보정만으로는 정상·난독화를 가르지 못함을 확인했다.
  다음 단계는 난독화 위치 탐지를 함께 학습하는 방향을 검토 중이다.
