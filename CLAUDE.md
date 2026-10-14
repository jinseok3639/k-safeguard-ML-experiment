# k-safeguard ML 실험 저장소 — 작업 규칙

자세한 설명은 `NOTES.md`(노트·리뷰·이름 규칙)와 `README.md`(MLflow 구조·보는 법). 여기는 매번
지켜야 하는 것만 모은다.

## 실험 흐름 (반드시 이 순서)

1. 단계 만들기 — 제목 필수
   `python -m <트랙>.exp.stage new sNN-<영문요약> --title "한국어 제목" --runs sNN.1-<영문요약> ... --run-titles "제목" ...`
2. 실행 **전**: `STAGE.md`의 **질문**, 각 run 노트의 **왜 돌리나·무엇을 바꿨나**(예상과 결과별 대응 포함)
3. 러너 실행: `python -m <트랙>.exp.run_... --run-id <run-id>`
4. 실행 **후**: run 노트의 **해석·다음**, `STAGE.md`의 **답·근거·다음 단계 제안**
5. `python -m <트랙>.exp.stage sync <단계-id>` — MLflow와 `EXPERIMENTS.md` 단계 표를 갱신
6. 사용자에게 리뷰 요청. 승인 전에는 같은 트랙의 다음 단계를 시작하지 않는다

## 브랜치

- 단계 하나 = 브랜치 하나 = PR 하나: `exp/sNN-<영문요약>`(단계 id와 같게). 이전 단계가 master에 merge된
  뒤 master에서 만든다.
- 여러 단계가 같이 쓰는 도구 수정(`kslab/`, MLflow 구성 등)은 `feat/…`·`fix/…` 브랜치로 따로 PR을 올려
  먼저 merge한다. 실험 중 꼭 필요하면 실험 브랜치에 넣고 PR 본문에 적는다.
- 러너와 실행 전 노트는 **커밋한 뒤** 실행한다(manifest에 커밋 해시가 남는다).
- 단계 승인 기록을 커밋한 뒤 PR → merge. merge는 **merge 커밋**으로 한다. squash·rebase 금지 — 결과에
  남은 커밋 해시가 master 이력에서 사라진다.
- 트랙이 다르면(복원기·가드레일) 실험 브랜치를 동시에 열어도 된다.

## 이름

- 단계 id `sNN-<영문요약>`, run id `sNN.k-<영문요약>`. 날짜를 넣지 않는다. id는 정한 뒤 바꾸지 않는다.
- 보이는 이름은 한국어 제목: `S03 기법별 임계값`, `S03.1 임계값 스윕`. 바꿀 때는 `stage title`.
- `e11b`처럼 뜻 없는 기호를 제목에 쓰지 않는다.

## 기록

- `STAGE.md`는 결론 먼저: 질문 → 답(한두 줄) → 근거(숫자 3~5개, `질문 | 기준 | 결과 | 판정` 표)
  → 다음 단계 제안 → 세부 해석 → 리뷰 기록.
- 러너는 `finish(metrics, headline=...)`에 대표 지표를 직접 넘긴다. 세부 평가표는 `evaluations`·`models`
  키 아래에 둔다 — MLflow에는 요약만 올라가고 전체는 `metrics.json`에 남는다.
- 모델을 여러 개 학습하는 러너는 `with tracking.fit_label(f"{label}/seed{seed}"):`로 감싼다.
  MLflow에서 모델별 run 이름이 된다.
- 리뷰 보고는 결정 규칙에 걸린 숫자 몇 개와 판정만. 표 더미를 그대로 넘기지 않는다.

## 하지 말 것

- 리뷰(`stage review --feedback`)를 사용자가 실제로 한 말 없이 기록하지 않는다. 스스로 승인하지 않는다.
- 결론을 낼 run에 `smoke`·`trial`·`scratch` id를 쓰지 않는다(게이트를 건너뛴다).
- 실행 전에 하지 않은 예상을 나중에 "예상했다"고 쓰지 않는다. 실행 전 칸을 실행 후 고치지 않는다.
- 공격 원문(`predictions.jsonl`, `run.log`)을 MLflow나 git에 올리지 않는다.
- 큰 모델 로드·사전 실측은 사용자 승인 후에만. `../k-safeguard` 저장소는 읽기 전용.
- `data/mlflow/`를 손으로 지우지 않는다. 구성을 바꾸면 `python -m kslab.mlflow_rebuild`(기존 저장소는 백업으로 옮김).
