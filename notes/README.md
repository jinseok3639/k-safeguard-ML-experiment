# 실험 노트와 단계 리뷰

수치는 MLflow와 `results/`가 남긴다. 여기는 **왜 돌렸는지, 어떻게 읽었는지, 그래서 무엇을
하기로 했는지**를 남기는 곳이고, 리뷰어(사용자)의 피드백 없이는 다음 실험으로 넘어가지
못하게 막는 곳이다. 규칙은 `kslab/notes.py`가 강제한다.

## 구조

```
notes/
  <단계-id>/
    STAGE.md        단계의 목적 · 종합 해석 · 다음 단계 제안 · 리뷰 기록
    stage.json      run 목록, 봉인 시각, 나중에 추가된 run, 리뷰 (기계가 읽는다)
    <run-id>.md     run 하나의 노트: 왜 돌리나 / 무엇을 바꿨나 / 해석 / 다음
```

**단계**는 함께 계획한 run의 묶음이고 리뷰의 단위다. 한 단계의 run은 병렬로 돌려도 되고,
리뷰는 단계당 한 번 받는다.

## 흐름

| 순서 | 하는 일 | 명령 |
|---|---|---|
| 1 | 단계와 노트 틀 생성 | `python -m exp.stage new <단계-id> --runs <run-id>...` |
| 2 | `STAGE.md`의 목적, 각 노트의 **왜 돌리나·무엇을 바꿨나** 작성 | (실행 전) |
| 3 | 러너 실행 — 같은 단계는 병렬 가능 | `python -m exp.run_... --run-id <run-id>` |
| 4 | 각 노트의 **해석·다음**, `STAGE.md`의 종합 해석·다음 단계 제안 작성 | (실행 후) |
| 5 | 리뷰 자료 확인 | `python -m exp.stage show <단계-id>` |
| 6 | 리뷰어 피드백 기록 | `python -m exp.stage review <단계-id> --verdict approve\|revise --feedback "..."` |

`python -m exp.stage status`로 전체 상태와 지금 무엇이 막고 있는지 본다.

## 강제되는 것

- **노트 없는 run은 시작하지 못한다.** "왜 돌리나"·"무엇을 바꿨나"가 비어 있어도 마찬가지다.
- **시작된 단계가 승인되기 전에는 다른 단계의 run이 시작되지 않는다.** `revise`는 피드백만
  남기고 단계를 열어 둔다 — 반영한 뒤 다시 리뷰를 받는다. `approve`만 단계를 닫는다.
- **리뷰를 기록하려면** 모든 run 노트의 해석·다음과 `STAGE.md`의 종합 해석·다음 단계 제안이
  채워져 있어야 하고, 피드백이 비어 있으면 안 된다.
- **승인된 단계에는 run을 더 넣을 수 없다.** 새 단계를 만든다.

## 드러나게 해 둔 것

- **실행 전 칸을 나중에 고치면 표시된다.** 러너가 시작할 때 노트 사본을
  `results/<run-id>/note-at-start.md`로 뜨고 실행 전 두 칸의 해시를 manifest와 MLflow 태그
  (`note.prereg_sha256`)에 남긴다. `show`가 지금 노트와 견줘 바뀌었으면 경고한다.
- **단계가 시작된 뒤 추가된 run은 따로 표시된다.** 추가는 가능하지만(`add-run --reason`)
  사유와 그 시점에 이미 결과가 나와 있던 run이 `stage.json`에 남고 `show`에 나온다.

## 지켜야 하는 것 (기계가 못 막는 부분)

- `review`의 `--feedback`은 **리뷰어의 말을 그대로** 적는다. 실험을 돌린 쪽이 스스로 승인하지
  않는다.
- run-id에 `smoke`·`trial`·`scratch`가 들어가면 게이트를 건너뛴다. 파이프라인이 끝까지 도는지
  보는 작은 실행에만 쓴다. 결론을 낼 실행에 쓰지 않는다.
- 실패한 run을 다시 돌릴 때는 새 run-id로 `add-run`하고 사유를 적는다.

## MLflow와의 관계

원본은 git에 커밋되는 이 폴더다(MLflow 저장소는 커밋되지 않는다). 러너가 시작할 때 노트가
MLflow run 설명으로 들어가고, `review`나 `python -m exp.stage sync <단계-id>`를 돌리면 해석과
리뷰 결과가 반영된다. 태그: `kslab.stage`, `kslab.compares_to`, `note.interpreted`,
`review.status`(pending/revise/approved), `review.feedback`.
