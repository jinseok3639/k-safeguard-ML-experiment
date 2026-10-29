# 실험 대장 — 가드레일 트랙

실패한 실험도 지우지 않는다. 가드레일 후보와 선정 기준은 [`CANDIDATES.md`](./CANDIDATES.md).
실험별 코드·기준값·원자료 위치와 이름 대응은 [`INDEX.md`](./INDEX.md), 이름 짓는 규칙은 [`NAMING.md`](./NAMING.md).

## 단계별 요약

단계(질문 하나) 하나에 한 줄. `python -m guardrail.exp.stage sync <단계-id>`가 다시 만든다.

<!-- stages:begin (stage sync가 다시 쓴다. 손으로 고치지 않는다) -->
| 단계 | 질문 | 답 | 상태 |
|---|---|---|---|
| **S01 난독화 강건성** | kslab 평가셋 v1의 난독화 16유형 중 어느 것이 인젝션 가드레일을 실제로 통과시키는가. | Kanana는 원문 공격을 가장 잘 막지만(94.0%) 난독화에 가장 많이 뚫린다(회피/원문차단 27.5%). Horizon 인코더는 14~15%로 덜 뚫리지만 난독화된 정상 문장 오차단이 원문의 2.4~6.5배로 오른다. 공통 최상위 위협은 `phon_insert`(받침 채우기·반모음 추가)다. | 실행 중 |
<!-- stages:end -->

## run 기록

| 날짜 | run-id | 실험 | 핵심 수치 | 상태 | 보고서 |
|---|---|---|---|---|---|
| 2026-10-27 | `s01.1-kanana` | S01.1 Kanana prompt 2.1b | 원문 탐지 94.0%, 회피\|원문차단 27.5%, 난독화 오차단 2.8% | `PROVISIONAL_DEV_ONLY` | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
| 2026-10-27 | `s01.2-horizon-base` | S01.2 Horizon base | 원문 탐지 90.7%, 회피\|원문차단 13.8%, 난독화 오차단 8.4%(원문 1.3%) | `PROVISIONAL_DEV_ONLY` | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
| 2026-10-27 | `s01.3-horizon-large` | S01.3 Horizon large | 원문 탐지 92.4%, 회피\|원문차단 14.6%, 난독화 오차단 6.8% | `PROVISIONAL_DEV_ONLY` | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
| 2026-10-27 | `s01.4-klue-roberta` | S01.4 KLUE RoBERTa | 원문 오차단 26.6%로 비교 제외(참고), 회피\|원문차단 42.3% | `DO_NOT_PROMOTE` | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
| 2026-10-27 | `s01.5-gemma4-e2b` | S01.5 Gemma 4 E2B(생성 판정) | 중단: 생성 방식 5.8~11.4행/초로 1시간 이상 예상. 결과 없음 | 중단 | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
| 2026-10-27 | `s01.5b-gemma4-e2b-logit` | S01.6 Gemma 4 E2B(확률 판정) | 중단: VRAM 초과·속도 저하로 2회 중단(83%에서 종료). 결과 없음 | 중단 | [S01](./reports/S01_OBF_ROBUSTNESS.md) |
