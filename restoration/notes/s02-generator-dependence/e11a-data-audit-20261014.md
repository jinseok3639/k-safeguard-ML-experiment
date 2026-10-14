# e11a-data-audit-20261014

- 단계: `s02-generator-dependence`
- 비교 대상: 없음 (데이터 구축)
- 실행 명령: `python -m restoration.exp.run_e11_generator_dependence --run-id e11a-data-audit-20261014 --part a`

## 왜 돌리나
e11b~d가 쓸 평가 데이터를 만들고 감사한다. 학습은 없다.
- KOTOX 라벨 코드(`1-1`, `8-3`, `13-2` 등)가 논문 번호와 맞지 않아, KOTOX 저장소
  `augmentation.py`의 `MAP`에서 대응을 확인해 넣었다(커밋 a01c917).
- 규칙을 slot(자모 단위) / char_swap / lexical / structural로 나누고, 쌍마다 길이 보존·슬롯
  가역 여부를 표시한다.
- benign: dev = KOTOX 학습 분할 정상 문장, test = KOTOX 테스트 분할 정상 문장 + 505 시드 benign.

- 예상: **실행 전 예상을 따로 세우지 않았다.** 파이프라인 점검용 스모크(`smoke-e11a`)를 먼저 돌려
  같은 결정론적 결과를 이미 봤다 — KOTOX 테스트 쌍 920개 중 슬롯 가역 22.3%, 규칙이 전부 slot인
  쌍 16.1%, benign dev 1,835 / test 434. 이 run은 그 결과를 정식 기록으로 남기는 것이다.

## 무엇을 바꿨나
새 러너 `restoration/exp/run_e11_generator_dependence.py --part a`. 산출 데이터는
`data/derived/s02/`(커밋 제외)에 쓰고 SHA-256을 manifest·MLflow에 남긴다.
입력: `data/hf-survey/ssgyejin__KOTOX/data/`(train 5,505 · easy 230 · hard 230행, MIT).

## 해석
스모크와 같은 결과가 나왔다(결정론적). `DATA_READY`.
- KOTOX test 920쌍 중 슬롯 가역 22.3%, 규칙이 전부 slot인 쌍 16.1%, lexical 포함 45.0%.
  train 11,010쌍은 각각 19.1% / 14.5% / 45.9%. 평균 CER이 0.49~0.51로 매우 높다 — 한 문장에
  규칙 2~4개가 겹쳐 글자 절반이 바뀐 수준이다. 우리 505 시드 평가셋보다 훨씬 무겁다.
- 길이 보존 쌍이 37~38%뿐이다. 우리 복원기는 길이를 바꾸지 못하므로, KOTOX 쌍의 3분의 2 가까이는
  구조적으로 일부만 고칠 수 있다. e11c 수치의 상한이 낮게 깔리는 이유다.
- benign dev 1,835 / test 434, 겹침 0. dev는 KOTOX 댓글체라 505 시드 benign(대화형 명령문)과
  문체가 다르다. dev에서 고른 임계값이 시드 benign에서 얼마나 맞는지는 e11b의 test 열로 본다.
- 믿을 수 있는 것: 규칙 코드 대응(저장소 원본 확인), 분할 겹침 0. 믿기 어려운 것: '슬롯 가역'은
  자모 단위 정렬로 판정한 근사다.

## 다음
e11b~d가 이 데이터를 쓴다. KOTOX 원문은 평가 전용으로 두고 학습에 넣지 않는다는 규약을 유지한다.
lexical(8-x)·structural(10·11·13-2)은 슬롯 복원기 범위 밖이라, 다음 단계에서 KOTOX 규칙을 이식한다면
slot 규칙(1-1·1-3·1-4·1-5·2-1·2-2·2-3·3-1·5-2)만 대상이다.
