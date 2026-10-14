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
<!-- 실행 후에 쓴다. 가설이 맞았나. 믿을 수 있는 것과 없는 것. -->

## 다음
<!-- 실행 후에 쓴다. 이 결과로 정해진 다음 실험, 그리고 버린 선택지. -->
