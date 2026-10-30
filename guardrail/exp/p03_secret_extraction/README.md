# P03 Tensor Trust 비밀값 추출 — 실행 안내

Tensor Trust(ICLR 2024) 추출 벤치마크의 공격을 우리가 쓰는 LLM과 가드레일에 그대로 넣어, 시스템 프롬프트 속
비밀번호가 새는지 잰다. 같은 샘플을 원문(`en`)과 한국어 번역본(`ko`)으로 각각 돌린다.

일은 두 갈래로 나뉜다.

| 하는 일 | 어디서 | 누가 |
|---|---|---|
| LLM 응답 생성 (`generate_llm.py`) | Colab GPU | 생성 담당 — 이 문서의 1~3 |
| 판정·기록 (`run_llm.py`, `run_guardrail.py`, `guarded.py`) | 로컬 | 실험 담당 — 이 문서의 4 |

생성 스크립트는 응답만 파일로 쓰고 판정하지 않는다. 판정과 기록은 실험 노트 게이트를 거쳐야 해서 실험 담당이
받아 온 파일로 따로 한다.

## 1. 준비 (Colab)

GPU 런타임에서 저장소를 받고 이 브랜치로 옮긴다. vLLM이 필요하다.

```bash
git clone https://github.com/jinseok3639/k-safeguard-ML-experiment
cd k-safeguard-ML-experiment
git checkout experiment/p03-secret-extraction
pip install vllm
```

데이터는 저장소에 들어 있지 않다(데이터에 라이선스 표기가 없다). 아래 경로에 둔다. 경로는 저장소 루트 기준이다.

```
data/ext/tensor-trust-data/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl        # en
data/ext/tensor-trust-data-ko/benchmarks/extraction-robustness/v1/extraction_robustness_dataset_ko.jsonl  # ko
```

- `en`은 공개 저장소를 정해진 커밋으로 받으면 된다. 전달받은 파일을 같은 경로에 둬도 된다.

  ```bash
  git clone https://github.com/HumanCompatibleAI/tensor-trust-data data/ext/tensor-trust-data
  git -C data/ext/tensor-trust-data checkout 747a75e096761ebc01bd3970158827326b4add23
  ```

- `ko`는 공개 저장소가 없다. 전달받은 파일을 위 경로에 둔다.
- 파일을 고치지 않는다. 스크립트가 sha256을 확인하고, 다르면 멈춘다.

제대로 놓였는지는 테스트로 확인한다. 실제 데이터 테스트가 skip 없이 통과하면 된다.

```bash
python -m unittest tests.test_p03_secret_extraction -v
```

## 2. 응답 생성 (Colab)

모델 하나 × 데이터 하나마다 한 번 돌린다. 모델과 리비전은 실험 담당이 정해 준 것을 쓴다. 아래 예시는 S02에서 쓴 두
모델이다.

```bash
python -u -m guardrail.exp.p03_secret_extraction.generate_llm --data en \
    --model LGAI-EXAONE/EXAONE-4.0-32B-AWQ --revision 1ce84bc \
    --out /content/drive/MyDrive/p03/exaone4-32b-awq-en.gen.jsonl

python -u -m guardrail.exp.p03_secret_extraction.generate_llm --data ko \
    --model kakaocorp/kanana-1.5-8b-instruct-2505 --revision c963a5f \
    --out /content/drive/MyDrive/p03/kanana15-8b-ko.gen.jsonl
```

- **출력 파일 이름은 `<모델>-<데이터>.gen.jsonl`로 하고, 모델·데이터마다 따로 둔다.** en과 ko는 행 id가 같아서 한
  파일에 섞이면 안 된다. 섞으려 하면 스크립트가 멈춘다.
- 출력은 Drive에 둔다. 세션이 끊기면 같은 명령을 다시 돌리면 쓴 행은 건너뛰고 이어서 한다.
- 처음 한 번은 `--limit 2`로 파이프라인만 확인한다. 이때는 출력 파일을 본 실행과 다른 이름으로 둔다.
- 디코딩(greedy), 생성 길이, 메시지 구성, 추론 모드 끄기는 스크립트에 고정돼 있다. `--data`, `--model`,
  `--revision`, `--out` 말고는 바꾸지 않는다. 메모리가 모자라 다른 옵션을 바꿔야 하면 바꾼 값을 같이 알려 준다.

샘플 569개에서 공격 입력과 비밀번호 입력을 하나씩 만들어 파일 하나에 1138행이 생긴다.

## 3. 돌려줄 것

- 만든 `.gen.jsonl` 파일들. 열어서 고치지 않는다.
- 마지막에 찍힌 `완료:` 줄(행 수, 걸린 시간, GPU 이름).

응답 파일에는 공격문과 비밀번호가 들어 있다. 저장소에 커밋하거나 공개된 곳에 올리지 않는다.

## 4. 판정·기록 (로컬, 실험 담당)

run id는 [`guardrail/NAMING.md`](../../NAMING.md) 규칙을 따르고, 돌리기 전에 실험 노트에 등록돼 있어야 한다.
각 명령에 그 run이 쓴 데이터를 `--data`로 준다.

```bash
python -u -m guardrail.exp.p03_secret_extraction.run_llm --data en --run-id <run id> \
    --generations <받아 온 .gen.jsonl>
python -u -m guardrail.exp.p03_secret_extraction.run_guardrail --data en --run-id <run id> --guardrail kanana
python -u -m guardrail.exp.p03_secret_extraction.guarded --guard <가드레일 run id...> --llm <LLM run id...>
```

`run_llm.py`는 생성 파일이 지금 데이터의 id를 빠짐없이 한 번씩 담았는지, `--data`와 같은 데이터로 만들었는지
확인하고 판정한다. `guarded.py`는 넘긴 run이 모두 같은 데이터일 때만 합친다.

## 문제가 생기면

| 메시지 | 뜻 |
|---|---|
| `Tensor Trust 데이터(…)가 없습니다` | 1의 경로에 파일이 없다 |
| `데이터 해시가 기록과 다릅니다` | 다른 버전이거나 고친 파일이다. 전달받은 원본으로 바꾼다 |
| `다른 데이터(…)로 만든 행이 있습니다` | 출력 파일을 다른 `--data` 실행과 같이 쓰고 있다. 파일 이름을 바꾼다 |
| `생성 파일이 데이터와 맞지 않습니다` | 생성이 중간에 끝났다. 같은 명령으로 이어서 마친다 |
