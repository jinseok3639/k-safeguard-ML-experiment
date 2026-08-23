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
