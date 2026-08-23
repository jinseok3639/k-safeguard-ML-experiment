# 실험 대장

실험 하나당 한 줄. 실패한 실험도 지우지 않는다 — 저장소의 `TENSIFY_LOCKED_RESULT.md`
선례대로, 안 되는 것을 기록해 두는 것도 결과다.

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
