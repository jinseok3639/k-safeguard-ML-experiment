# kslab — k-safeguard ML 실험

[k-safeguard](https://github.com/jinseok3639/k-safeguard)에서 "규칙으로 답이 안 나오는 난독화
케이스에 ML을 붙일 수 있는가"를 실증하는 실험 저장소다. 라이브러리에 들어가는 결정의 근거가
여기에 있다. 실험은 두 트랙으로 나뉜다.

| 트랙 | 묻는 것 |
|---|---|
| [`restoration/`](./restoration/README.md) | 복원기(난독화 → 원문 후보)를 얼마나 잘 만들 수 있나 |
| [`guardrail/`](./guardrail/README.md) | 어떤 가드레일로, 어떤 난독화가 실제로 위협인지 재나 |

두 트랙이 함께 쓰는 데이터 조사와 사용 결정은 [`DATASETS.md`](./DATASETS.md)에 있다.

이 저장소의 문서에는 바뀌지 않는 원칙과 결론만 적는다. 모듈 구성, 함수, 파일 위치, 명령 옵션처럼 코드가 정하는
것은 각 모듈의 docstring과 `--help`가 원본이다.

## 결과 읽는 법

- **트랙별 실험 대장** `<트랙>/EXPERIMENTS.md` — 단계별 질문과 답, run별 핵심 수치와 판정.
- **보고서** `<트랙>/reports/` — 실험별 상세.

## 실험 원칙

- **평가 데이터는 학습에 쓰지 않는다.** 505개 시드, 평가 원문, 외부 생성기 문장은 평가에만 쓴다.
- **후보만 만들고 판단하지 않는다.** block/allow를 반환하는 코드는 이 저장소에 두지 않는다.
- **k-safeguard 저장소는 읽기 전용으로 참조한다.**
- **실행 전에 질문과 판정 기준을 적고, 결과를 본 뒤 고치지 않는다.** 고치면 드러나게 기록이 남는다.
- **공격 원문이 든 산출물은 공개하지 않는다.** run 산출물과 데이터는 커밋하지 않고, 공개하는 것은 집계와 보고서뿐이다.
- **실험 결과의 원본은 run 결과 폴더와 노트다.** MLflow는 그것으로 언제든 다시 만들 수 있는 사본이다.
- **실험의 흔적은 저장소 폴더 안에만 남긴다.** 모델·데이터 캐시와 기록 저장소를 사용자 홈 등 밖에 두지 않는다.

## 시작하기

```bash
. .\env.ps1                 # Windows PowerShell. bash에서는 source ./env.sh
python -m unittest discover -s tests
python -m restoration.exp.stage status
```

러너는 `python -m <트랙>.exp.<러너>`로 실행한다.
