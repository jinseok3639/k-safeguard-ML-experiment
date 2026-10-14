# e11d-oracle-detector-20261014

- 단계: `s02-generator-dependence`
- 비교 대상: `e8-mlp-bilstm-30k-20261007` (단일 기법 모델)
- 실행 명령: `python -m restoration.exp.run_e11_generator_dependence --run-id e11d-oracle-detector-20261014 --part d --no-progress`

## 왜 돌리나
탐지기·수정기를 나누는 구조가 값어치 있는지 미리 가른다. 손상된 자리를 정확히 알려주면(완벽한
탐지기) 모델이 얼마나 고치는지(oracle)와, benign dev 1% 임계값에서의 실제 성능(actual)을 비교한다.

- 예상(실행 전, Claude): 된소리는 oracle이 2%p 기준을 넘고 actual과의 차이(탐지 손실 몫)가
  작을 것이다 — 경음 초성이라는 흔적 덕에 탐지가 이미 쉽다. 연음은 oracle도 기준 근처이거나
  미달일 수 있다 — 받침이 어디서 넘어왔는지 교정 자체가 어렵다. 받침 크래밍은 탐지 손실 몫이
  클 것이다(정상 받침과 구별이 어려움).
- 결과별 대응: oracle도 기준 미달인 기법은 탐지기를 붙여도 소용없으니 분리 구조 검토 대상에서
  뺀다. oracle은 넘고 탐지 손실 몫이 큰 기법이 분리 구조의 대상이 된다.

## 무엇을 바꿨나
- 모델 두 종류: (1) e11b의 `all` 합집합 모델 3개(체크포인트 재사용) — 기법군별로 봄,
  (2) 단일 기법 BiLSTM(된소리·연음·받침 크래밍) × 시드 3 = 9회 새로 학습. 단일 모델은 E8b와 같은
  데이터 구성(코퍼스 학습 분할 전체 × identity + 강도 0.5·1.0).
- oracle은 원문과 다른 자리만 모델 argmax로 바꾼다. benign 변형률은 정의상 0이다.
- 평가: 505 시드에 기법별 강도 0.5·1.0.

## 해석
상태 `MEASURED`. 예상은 반만 맞았고, **'탐지 손실'의 정체가 예상과 다르다.**
- oracle 2%p 통과: 합집합 모델 기준 tensify·liaison·coda·vowel 통과, assimilation(+0.44) 미달.
  기법 단위로는 tensify_rule(+0.43)·cluster_reduce(+0.23)·nasalize(+0.49)·coda_neutralize(+1.33)·
  aspirate(+1.30)가 oracle로도 미달 — 이건 복원기 문제가 아니라 손상량 자체가 작다(발동률·변형 폭).
- 예상 대조: tensify 탐지 손실 몫이 작을 것 → **틀림**(합집합 46%, 단일 64%). liaison oracle이 기준
  근처·미달 → **틀림**(+5.02/+6.51%p로 통과). 받침 크래밍 탐지 손실 큼 → 맞음(단일 69%, 합집합은 기법
  단위 99%).
- **핵심 관찰**: abstention 없음(모든 예측 적용)의 공격 쪽 CER 감소가 oracle과 거의 같다 — 기법 단위로
  tensify 19.43 vs 19.78, jongseong_cram 33.15 vs 34.83, vowel_confuse 14.11 vs 14.88, liaison 4.61 vs
  5.04(합집합 모델, %p). 즉 모델은 **공격 문장에서 손상 자리를 이미 거의 정확히 찾는다.** 잃는 몫은
  '손상 자리를 못 찾아서'가 아니라 'benign 1%를 지키려고 올린 임계값이 손상 자리 수정까지 잘라서'다.
  리포트의 '탐지 손실 몫'은 '임계값 비용'으로 읽어야 정확하다.
- **합집합 모델의 공통 임계값이 받침 크래밍을 죽인다**: jongseong_cram 기법 단위로 합집합 actual
  +0.21%p, abstention 없음 +33.15%p, 단일 모델 actual +12.08%p. 합집합 모델도 고칠 능력은 있는데,
  임계값 하나(≈0.9993~0.9997)를 tensify처럼 확신이 높은 기법에 맞추니 크래밍 수정이 전부 잘린다.
  e11b의 coda 분모가 0에 가까웠던 이유도 이것이다.
- 비교 주의: 리포트 표의 `all` 행은 기법군 평균, `single_*` 행은 기법 하나다. 같은 기법으로 맞추면
  tensify는 합집합 +10.72%p > 단일 +7.83%p(합집합이 낫다), liaison은 2.25 < 3.75, cram은 0.21 ≪ 12.08.
- 단일 모델 일부는 benign test 변형률이 1%를 넘는다(single_tensify seed2026 1.61%, single_liaison
  seed2027 1.38%) — dev에서 고른 임계값이 test로 완전히는 안 넘어간다.

## 다음
- 탐지기·수정기 분리의 동기는 '공격 문장에서 손상 자리 찾기'가 아니라 **'benign과 공격을 가르는
  보정'**이다. 다음 실험 후보: (1) 학습 없이 기존 체크포인트로 기법(예측 편집 종류)별·슬롯별 임계값을
  benign dev에서 따로 고르기 — 비용이 거의 없고 cram 회복 폭을 바로 잰다. (2) 그래도 부족하면 benign
  음성 예시를 넣은 문장·자리 단위 탐지기.
- oracle로도 2%p 미달인 기법(tensify_rule·cluster_reduce·nasalize·aspirate·coda_neutralize·lateralize·
  h_deletion·palatalize — 모두 oracle +1.33%p 이하)은 분리 구조 대상에서 뺀다. 생성기 쪽에서 손상량이 의미 있는지부터 봐야 한다.
- 확정은 리뷰에서.
