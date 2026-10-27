# kslab — 공유 라이브러리

두 실험 트랙([`restoration/`](../restoration/README.md), [`guardrail/`](../guardrail/README.md))이
함께 쓰는 코드다. 자모 처리, 난독화 분류표와 생성, 평가셋 구성, 학습 데이터 구성, 슬롯 복원 모델,
지표, run 기록과 리뷰 게이트가 여기 있다. 러너(`<트랙>/exp/run_*.py`)는 이 패키지를 조립할 뿐 자체 로직을 거의
갖지 않는다.

이 패키지는 이 실험 저장소 전용이다. k-safeguard 저장소는 읽기 전용으로만 참조하고, 그쪽에는
아무것도 쓰지 않는다.

## 데이터 흐름

```
 평가 원문 695 (평가 전용)                                   위키 + KLUE 코퍼스 (corpus.py, 학습용)
 = 505 시드(hf_repo/seeds) + 공개 정상 문장 190                        │
        │                                                             │
        │      taxonomy.py   통합 분류표 — 기법 → 유형 → 복원기 범위      │
        │                                                             │
        ├──▶ evalset.py + recipe.py + external_gen.py                 │
        │      평가셋 v1(축: single·compose·external·control),         │
        │      문장마다 레시피                                          │
        │                                    obfuscate.py / compose.py  — 난독화를 건다
        │                                                             │
        └──────────────── pairs.py ──────────────────────────────────┘  (난독화, 원문) 페어 · 분할 · 누출 검사
                              │
                        features.py      후보 위치 + 슬롯 특징
                              │
          gate.py ─▶   models/           linear → mlp → bilstm, decision(된소리 판단)
                              │
               ceiling.py · metrics.py    구조적 상한 · proxy 지표 · bootstrap
                              │
         record.py ─▶ tracking.py        manifest · baseline 발행 · MLflow
                              │
                     provider.py          k-safeguard CandidateProvider 계약에 맞춘 프로토타입
```

## 모듈

### 한글 기초

| 모듈 | 내용 |
|---|---|
| `jamo.py` | 완성형 음절 ↔ (초성, 중성, 종성) 슬롯의 무손실 변환. 테이블은 k-safeguard `normalization.py:18-34`의 값을 복제했고 테스트가 원본과 대조한다 |
| `phonology.py` | 표준발음법 규칙 테이블(격음화·비음화·유음화·평파열음화·자음군 단순화·경음화)과 혼동 모음. 표만 선언하고 변환은 `obfuscate.py`가 한다 |

### 난독화 생성

| 모듈 | 내용 |
|---|---|
| `obfuscate.py` | 단일 난독화 24종. 규약은 `(text, intensity=1.0, seed=0) -> str`. 레지스트리 `TRANSFORMS`와 분류 상수(`FAMILY`, `LENGTH_PRESERVING`, `MODELED`, `STANDARD_PRONUNCIATION_CLAUSE`) |
| `compose.py` | 다중(조합) 난독화. 정준 순서(`CANONICAL_ORDER`), 단계별 시드 분리, 손상량을 고정한 채 깊이만 바꾸는 강도 보정(`calibrate`), 자리별 출처 추적. 이름 붙인 조합은 `PROFILES` |
| `taxonomy.py` | 통합 난독화 분류표 16유형. KOTOX(arXiv 2510.10961)의 5분류(음운·모양·음차·구조·화용)를 뼈대로 하고 KOTOX 밖 유형(자모 분해, 음절 늘이기, 초성체)을 더했다. 유형마다 KOTOX 규칙 코드, 기법, 복원기 범위(`target`·`boundary`·`out`)를 적는다. 모든 기법은 정확히 한 유형에 속하고(`type_of`), 실험 결과는 이 표의 행으로 집계한다 |
| `recipe.py` | 난독화 레시피. 원문에 건 단계(생성기·기법·강도·시드) 목록과, 단계마다 바뀐 구간을 원문 위치로 되돌려 연결한 편집 기록. 같은 레시피를 같은 코드에서 다시 실행하면 같은 문장이 나온다(`run_internal`) |
| `external_gen.py` | 외부 난독화 생성기 래퍼 — KOTOX, kotka, korean-jamo-typo(모두 MIT). 저장소마다 `data/ext/<이름>/.venv`를 따로 두고, 우리 드라이버(`ext_drivers/`)를 그 파이썬으로 돌려 표준입력 JSONL로 문장을 넘긴다. 외부 기법 → 유형 대응(`TYPE_OF`)도 여기 있다. **학습에는 쓰지 않는다** |

기법 24종:

| 기법 | 계열 | 길이 보존 | 1기 슬롯 모델 대상 |
|---|---|:-:|:-:|
| `jamo_decompose`, `break_spacing`, `zwsp_inject` | visual | | |
| `symbol_insert`, `lookalike_foreign`, `lookalike_hangul` | visual | | |
| `chosung` | visual | ✓ | ✓ |
| `tensify`, `liaison`, `jongseong_cram` | phonetic | ✓ | ✓ |
| `vowel_confuse`, `aspirate` | phonetic | ✓ | ✓ |
| `palatalize`, `liaison_correct`, `h_deletion`, `coda_neutralize`, `cluster_reduce`, `nasalize`, `lateralize`, `tensify_rule` | phonetic | ✓ | |
| `semivowel_insert`, `initial_insert` | phonetic | ✓ | |
| `jamo_copy`, `syllable_extend` | phonetic | | |

- P01에서 새로 더한 7종. KOTOX·KoNTe·yu2024(NAACL Findings)에 있는데 기존 생성기에 없던 유형을 채운다: `semivowel_insert`(거품→궈퓸), `initial_insert`(많이→많휘), `jamo_copy`(쓰레기→쓸레기), `symbol_insert`(기호·이모티콘 끼워 넣기), `syllable_extend`(화장실→호와당실), `lookalike_foreign`(ㄱ→7, 가→㉮), `lookalike_hangul`(야민정음, 귀엽다→커엽다).
- 닮은 글자 두 기법은 KOTOX의 `iconic_dictionary.json`을 `data/ext/kotox-src`에서 읽는다(외부 생성기 준비 참고). 사전 오류 한 쌍은 `_KOTOX_ERRATA`로 뺀다.
- 기법이 어느 유형에 속하는지는 `taxonomy.type_of(기법)`로 본다.

- **visual**은 정규화로 되돌릴 수 있는 계열(대조군), **phonetic**은 정규화로 되돌릴 대상이 없는 음운 변형이다.
- 슬롯 분류는 길이 보존 기법에서만 성립한다. 그래서 1기 슬롯 모델용 조합(`compose.py`)에는 길이 보존이 아닌 기법과 `chosung`을 쓰지 않는다(`compose.NOT_COMPOSABLE`). 평가셋 복합 축(`evalset.py`)은 `chosung`만 빼고 모든 기법을 쓴다.
- 복원 모델 대상(`MODELED`)은 신호 밀도(실제 변형 자리 / 모델이 방문하는 후보 자리)로 골랐다. 나머지는 레드팀 데이터 생성기로만 쓴다.

### 데이터

| 모듈 | 내용 |
|---|---|
| `corpus.py` | 학습용 clean 문장 풀. 한국어 위키 + KLUE(모두 CC BY-SA 4.0)를 스트리밍해 필터링된 문장만 `data/corpus/`에 jsonl로 남긴다. 문서 단위 `source`로 해시 분할 |
| `pairs.py` | 505 시드 로드, (난독화, 원문) 페어 구성(`make_pairs`, `make_composed_pairs`), source-disjoint 분할과 누출 검사 |
| `evalset.py` | 평가셋 v1 정의. 원문 695(505 시드 + `xxxjjhhh/korean_guardrail_test`의 정상 문장 190, CC BY 4.0)를 불러오고, 단일·복합 기법 레시피를 만든다. 복합은 유형이 겹치지 않게 기법 2·3·4개를 고르며(`compose_for`), 사람형(KoNTe 관찰 건수 가중)과 자동형(균등) 공격자를 따로 둔다. 문장을 바꾸지 못한 기법은 빼고 다시 뽑는다. 발음 규칙 환경에서만 걸리는 기법(`SPARSE`)은 강도 1.0으로 건다. 빌더는 [`restoration/exp/run_p01_4_evalset_v1.py`](../restoration/exp/run_p01_4_evalset_v1.py), 산출물은 `data/derived/evalset_v1/`(커밋 제외) |
| `features.py` | 기법별 후보 위치 규칙과 "모르는 슬롯"(`TECHNIQUE_SPEC`, `UNKNOWN_SLOTS`), 슬롯 예제 추출, 문자 창·자모 창 인코딩. 조합 기법(`liaison+tensify`)은 구성 기법의 합집합으로 푼다 |

### 복원 모델

| 모듈 | 내용 |
|---|---|
| `models/linear.py` | 로지스틱 회귀 슬롯 복원기. 대조군이자 코퍼스 스케일링 실험의 고정 아키텍처 |
| `models/mlp.py` | 자모 임베딩 + 공유 몸통 MLP. 고정 폭 문자 창(±4자)을 본다 |
| `models/bilstm.py` | 소형 양방향 LSTM. 문장 전체를 읽는다. 체크포인트 재개 지원 |
| `models/decision.py` | 된소리 되돌림 판단 — 위치별 이진 분류 + abstention |
| `models/device.py` | 장치 선택. 기본은 CPU, 러너가 `--device`로 GPU를 켠다 |
| `gate.py` | 문자 n-gram(stupid backoff) 게이트. 어절 단위로 부자연스러운 곳만 복원 모델로 보낸다. 신경망 없이 결정론적 |

세 슬롯 복원기는 같은 인터페이스를 지켜 서로 바꿔 끼울 수 있다.

```python
model = MlpSlotRestorer(technique="liaison").fit(examples)    # examples: features.extract_examples(...)
model.predict_slots(examples)                                  # 슬롯별 예측
model.restore_many(texts, threshold=0.9)                       # [(복원문, 자리별 확신도), ...]
```

복원기는 `candidate_positions(obf, technique)`가 고른 자리만 방문하고, 그 자리에서도
`UNKNOWN_SLOTS[technique]`의 슬롯만 예측한다. 나머지는 입력을 그대로 복사한다. `threshold`
아래로 확신하는 자리는 건드리지 않는다.

### 평가

| 모듈 | 내용 |
|---|---|
| `ceiling.py` | 학습 없이 구하는 구조적 복원 상한. 손상 자리를 도달 불가(후보 규칙이 안 고름)와 표현 불가(담당 슬롯 밖)로 분해한다. 상한이 승격 기준에 못 미치면 학습할 이유가 없다 |
| `metrics.py` | 가드레일 없이 재는 proxy 지표 — clean mutation rate, 음절 exact, CER, 자리 단위 4분류. seed-cluster bootstrap은 k-safeguard 구현을 이식했다 |

### 실험 기록과 리뷰 게이트

| 모듈 | 내용 |
|---|---|
| `tracks.py` | 현재 트랙과 트랙별 경로(`results/`, `baselines/`, `reports/`, 노트, 실험 대장). 러너 패키지 `<트랙>/exp/__init__.py`가 트랙을 정한다 |
| `record.py` | `Run` — run-id 검사, 경로 격리 검사, manifest(git 커밋·dirty, 입력 SHA-256, 패키지 버전), baseline 발행 |
| `tracking.py` | MLflow 기록. `Run.start()`/`Run.finish()`가 부르므로 러너는 직접 쓰지 않는다 |
| `mlflow_rebuild.py` | MLflow 저장소를 원본(결과 폴더·노트)에서 다시 쓴다. `python -m kslab.mlflow_rebuild` |
| `notes.py` | 실험 노트와 단계 리뷰 게이트. 단계 id는 `s03-…`(1기)·`p01-…`(2기) 모두 받아 `S03`·`P01`로 표시한다. 노트의 실행 전 칸이 비면 시작을 거부하고, 시작 시점 노트를 해시와 함께 떠 둔다. 승인 전에는 다음 단계를 시작할 수 없다 |
| `stage_cli.py` | 단계 관리 CLI 본체. 각 트랙의 `exp/stage.py`가 부른다(`new`, `add-run`, `status`, `show`, `sync`, `review`) |

### k-safeguard 연결

| 모듈 | 내용 |
|---|---|
| `provider.py` | k-safeguard `CandidateProvider` 계약을 만족하는 프로토타입(`MlRestoreProvider`)과 계약 점검(`contract_report`). 저장소를 설치하지 않고 `src`를 `sys.path`에 얹어 실제 protocol을 쓰며, 바이트코드도 남기지 않는다 |

## 코드로 강제되는 규칙

테스트(`tests/`)와 런타임 검사가 아래를 지킨다. 바꾸려면 해당 테스트부터 고쳐야 한다.

- **505 시드는 평가 전용.** 학습 페어를 만드는 함수는 시드를 받지 않는다.
- **모든 기법은 분류표의 한 유형에 속한다.** 새 기법을 `TRANSFORMS`에 넣고 `taxonomy.py`에 등록하지 않으면 테스트가 실패한다.
- **레시피는 같은 문장을 다시 만든다.** 레시피로 다시 실행한 결과와 기록된 편집 구간이 원래 출력과 맞는지 테스트가 확인한다.
- **분할은 source-disjoint.** 같은 출처(템플릿·문서)의 문장이 train/test 양쪽에 들어가지 않는다. 랜덤 분할은 중성 슬롯 정확도를 15%p 부풀렸다.
- **후보 위치는 입력만 보고 정한다.** `candidate_positions()`는 난독화된 문자열만 받는다. 정답과의 차이를 후보로 쓰면 clean mutation이 실제보다 좋게 나온다.
- **길이 보존 선언은 505 시드 전체에서 검증된다.**
- **자모 테이블은 k-safeguard 원본과 같다.**
- **캐시는 저장소 안에 둔다.** `env.sh`/`env.ps1`을 거치지 않고 러너를 돌리면 `record.assert_paths_contained()`가 멈춘다.
- **run-id는 덮어쓰지 않는다.** run-id에 `smoke`·`trial`·`scratch`가 들어가면 스모크 실행으로 보고 baseline을 발행하지 않으며 MLflow에서도 따로 모은다.

## 러너에서 쓰는 법

러너의 골격은 대략 이렇다(실제 예: [`restoration/exp/run_e8_abstention.py`](../restoration/exp/run_e8_abstention.py)).

```python
from kslab import corpus as C, pairs as P, record as R

seeds = P.load_seeds()
run = R.Run.start(run_id=args.run_id, experiment="E8_abstention", config={...})
run.add_input("attacks.jsonl", P.SEED_DIR / "attacks.jsonl")

corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
run.add_input("corpus", corpus_path)
train_items, _ = C.split_by_hash(C.corpus_as_items(C.load_corpus(corpus_path)), test_fraction=0.2)
train_pairs = P.make_pairs(train_items, [technique], intensities, gen_seed=..., include_identity=True)
# ... 학습 → 시드로 평가 → metrics

run.write_text("report.md", report)
run.finish(metrics, headline={...})
R.publish_baseline("e8_abstention_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id)
```

## 외부 경로

- 505 시드는 이 저장소의 `hf_repo/seeds/`에 있다.
- k-safeguard 저장소는 이 저장소와 **같은 부모 폴더**에 `k-safeguard/`로 있어야 한다. `src/`(provider)와
  자모 테이블 대조 테스트가 거기서 읽는다.
- 코퍼스·모델 캐시와 평가셋은 저장소 안 `data/`에 쌓인다(커밋 제외).

## 외부 생성기 준비

`external_gen.py`를 쓰거나 평가셋을 다시 만들 때만 필요하다. 저장소마다 가상환경을 따로 둔다(의존성 버전이 서로 다르다).

| 이름 | 폴더(`data/ext/` 아래) | 출처 |
|---|---|---|
| `kotox` | `kotox-src` | https://github.com/leeyejin1231/KOTOX |
| `kotka` | `kotka` | https://github.com/Astro36/kotka |
| `jamo_typo` | `korean-jamo-typo` | https://github.com/SJLee0311/korean-jamo-typo |

```bash
git clone <출처> data/ext/<폴더>
python -m venv data/ext/<폴더>/.venv        # 그 저장소의 의존성을 이 venv에 설치
```

- 평가셋 v1에 쓴 각 저장소의 커밋은 `data/derived/evalset_v1/header.json`에 남는다. 같은 문장을 다시 만들려면 그 커밋으로 맞춘다.
- KOTOX의 GPT 기반 규칙(8-1, 8-3)은 쓰지 않는다. 1-5(소리 나는 대로 적기)는 G2P(`kotox-src/G2P`)가 있을 때만 켜진다. G2P의 형태소 분석기는 Java가 필요하며, 시스템 Java가 없으면 `data/ext/jdk`에 풀어 둔 JDK를 쓴다.

## 테스트

```bash
python -m unittest discover -s tests       # 저장소 루트에서
```
