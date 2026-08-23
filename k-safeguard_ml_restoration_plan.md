# 일대일 변환 불가 난독화 케이스 ML 복원 모델 설계

## Context

`src/k_safeguard/`의 정규화 미들웨어는 두 갈래다: 정보 손실이 없는 규칙(자모 조합, ZWSP 제거)은
core에서 무조건 적용하고, 정보가 소실돼 여러 원문으로 되돌릴 수 있는 변형(초성체, 된소리)은 opt-in
`CandidateProvider`가 사전(lexicon) 기반으로 제한된 후보를 만든다. 이 사전 기반 접근이
`experiments/benchmark/CHOSUNG_GUARDRAIL_IMPACT.md`(상태 `PROVISIONAL_DEV_ONLY`)에서 NRR
13.04%(목표 50% 미달)로 한계가 실측됐다 — 근본 원인은 "사전에 없는 단어는 원천적으로 후보를 만들 수
없다"는 lookup 아키텍처의 구조적 한계다. 연음·종성 크래밍은 `hf_repo/ko_obfuscator.py`에 생성기조차
없어 후보 provider가 아예 없다.

**이 설계의 범위**: 규칙/알고리즘으로 답이 나오지 않는 케이스에 한해 ML을 검토한다. 팀 합의나 결정
문서 갱신 같은 절차는 이 설계의 범위가 아니다 — 이건 기술적 견적(feasibility estimate)이다.

**ML의 의미**: 트랜스포머·어텐션·"SLM"에 한정하지 않는다. 아래 조사에서 보듯 어텐션이 전혀 없는
가벼운 분류 모델도 이 문제 유형에서 실적이 있다.

---

## 참고 사례: 유사 Dacon 대회 solution 3건 조사

`search/dacon/` 아래 1~3위 solution 노트북과 실제 `train.csv`(11,263행, 한국어 리뷰 텍스트에
음운적 노이즈를 적용한 (input, output) 쌍)를 직접 열어 확인했다.

### 태스크 성격 확인 — 이게 가장 중요한 발견이다

`train.csv` 2,000행을 표본 검사한 결과 input과 output의 **글자 수가 97%에서 완전히 같다**
(나머지 3%는 output 끝의 trailing space 하나 차이일 뿐, 실질적 삽입/삭제 없음). 예:

```
input:  쩡결학꼬 찐절하쉬넥오
output: 청결하고 친절하시네요
input:  았깝땀  (았: 종성 ㅆ 삽입, 땀: ㄷ→ㄸ 경음화 + 종성 ㅁ 삽입)
output: 아깝다
```

즉 이 대회의 "난독화"는 종성 삽입/치환(종성 크래밍과 직접 겹침), 평음·경음·격음 3중 혼동(된소리와
겹치는 부분집합), 모음 혼동(ㅔ/ㅐ 등, k-safeguard 용어집엔 없지만 같은 계열)으로 구성된
**음절 단위 자모 성분 치환 노이즈**이고, 전부 위치를 보존한다. k-safeguard가 다루려는 초성체·
연음·종성 크래밍도 구조적으로 같다 — 초성체도 "ㅇㄴ" 같은 호환 자모가 원래 음절 하나당 문자 하나를
그대로 차지하므로 길이가 안 변하고, 연음도 인접 음절 간 자모 재배치일 뿐 음절 수가 안 변한다(생성기가
아직 없어 최종 확인은 구현 후 필요).

**결론**: 이 복원 문제는 "새 문자열을 생성"하는 seq2seq가 아니라 **입력과 같은 길이의 출력을
위치별로 분류(classify)하는 문제**다. 이게 확인되면서 beam search나 encoder-decoder 없이도
설계가 훨씬 단순해진다.

> 이 세션에서 이 두 주장은 직접 재현·검증했다: `train.csv` 전체 11,263행을 다시 돌려 same-length
> 비율 97.09%를 재현했고(나머지도 전부 output 끝 공백 1개 차이), 1위 노트북 소스를 grep해
> `Embedding`→`MLP`(BatchNorm, RNN·attention 없음)→`Linear(hidden,68)` 1단계 구조와
> `monologg/koelectra-base-v3-discriminator`를 `ElectraForMaskedLM`으로 쓰는 2단계 구조를
> 확인했다.

### 3위 (`[Private_3위] 음절 단위 커스텀 토크나이저 + 다층 BiLSTM.ipynb`) — 어텐션 없음

- 음절(문자) 단위 커스텀 토크나이저(자모 분해 아님, `list(text)`로 직접 vocab 구성, vocab_size=2473).
- CBOW로 문자 임베딩을 먼저 비지도 사전학습 후 freeze.
- **4층 양방향 LSTM(어텐션 전혀 없음) + LayerNorm + Dropout**, 각 위치마다 vocab 전체에 대해
  softmax 분류 — 순수 position-wise sequence labeling.
- CrossEntropyLoss로 20 epoch 학습, train loss 0.10 → 0.003으로 수렴(과적합 위험은 있지만 이
  태스크가 이 정도로 가벼운 모델로도 학습 가능함을 보여준다).

### 1위 (`[private_1위] 읶겄돆 핶돆핶볶싞짂.ipynb`) — 2단계, 1단계는 MLP뿐

- **1단계("1차 해독")**: `Embedding`(자모 68종 + 위치 정보) → `MLP`(BatchNorm + skip
  connection) → `Linear(hidden, 68)` — 순수 position-wise 자모(초성/중성/종성 각각) 분류기.
  RNN도 attention도 없다. 문자 단위가 아니라 **자모 단위로 분해해 초성/중성/종성을 각각 별도
  클래스로 예측**하는 게 3위와의 핵심 차이 — 이건 chosung처럼 "초성은 이미 알고 있고 중성·종성만
  예측하면 되는" 부분 정보 상황에 훨씬 잘 맞는 프레이밍이다.
- **2단계("2차 해독")**: 1단계 출력을 사전학습된 한국어 ELECTRA(`monologg/koelectra-base-v3-discriminator`,
  약 1.1억 파라미터)로 `ElectraForMaskedLM` 방식으로 미세 보정. 여기서만 attention(pretrained
  encoder)이 등장하고, 그마저도 "밑바닥부터 학습하는 대형 모델"이 아니라 공개된 소형 한국어
  인코더를 그대로 fine-tune한 것이다.

### 2위 (`CSU_AI.ipynb`) — 참고만, 이번 제약과 안 맞음

Mistral-Small-24B, ko-gemma-2-9b, EEVE-Korean-Instruct-10.8B, DeepSeek-R1-Distill-Qwen-7B 등
**대형 LLM 여러 개를 파인튜닝해 앙상블 보팅**(초성/중성/종성 일치도로 후보 점수화)하는 방식.
정확도 상한은 가장 높을 가능성이 있지만 "로컬 소형 모델, 오프라인 CPU 추론"이라는 이번 서빙
제약과 정면으로 안 맞는다 — 이번 설계에서는 채택하지 않는다. 다만 "성능 상한을 더 밀어붙이고
싶을 때가 있다면 이 방향이 있다"는 참고 지점으로만 남긴다.

### 라이선스 주의 (미해결)

이 Dacon `train.csv`(`search/dacon/open.zip`)를 k-safeguard 학습 데이터로 실제 사용하려면
대회 데이터 라이선스(비공개 대회 데이터는 통상 재배포·타 프로젝트 전용이 금지됨)를 먼저 확인해야
한다. 이 설계에서는 **아키텍처·태스크 프레이밍의 참고 자료로만** 쓰고, k-safeguard 자체 데이터로
전제하지 않는다.

---

## 커버 범위 경계

| 포함 | 제외 |
|---|---|
| 초성체 OOV(사전 밖) 위치별 자모 복원 | 자유 의역/번역 |
| 된소리(tensify) 위치별 자모 복원 | 호모글리프, 코드스위칭 비한국어 구간 |
| 연음(liaison) 위치별 자모 복원 — 생성기 신규 필요 | 멀티모달, 에이전트형(도구호출) 위협 |
| 종성 크래밍 위치별 자모 복원 — 생성기 신규 필요 | 다중 턴 문맥 공격 |
| | **block/allow 최종 판단 — 항상 하위 가드레일** |

`CandidateProvider` protocol은 `generate() -> Iterable[CandidateProposal]`만 요구하고 판정
메서드가 없다 — 이 인터페이스 자체가 "ML은 후보만 만들고 판단은 하지 않는다"는 경계를 구조적으로
강제한다.

---

## 모델 아키텍처

### 태스크 재정의: 자모(jamo) 단위 위치별 분류

각 원문 음절을 초성·중성·종성 3개 슬롯으로 분해(`chosung.py`가 이미 가진 상수 재사용)하고, 각
슬롯을 68개 클래스(19 초성 + 21 중성 + 28 종성) 중 하나로 분류하는 문제로 정식화한다. 입력 슬롯
중 이미 확정된 값(예: 초성체는 초성이 이미 주어짐, tensify는 중성·종성이 이미 주어짐)은 입력
feature로 그대로 넣고, 모델은 **모르는 슬롯만** 예측하면 된다 — 이게 1위의 자모 단위 프레이밍이
chosung 같은 부분 정보 문제에 특히 잘 맞는 이유다. 길이가 보존되므로 beam search나
encoder-decoder가 필요 없다.

### 아키텍처 후보 비교 (전 스펙트럼)

| 후보 | 어텐션 | 파라미터 | 근거·판정 |
|---|---|---|---|
| **자모 임베딩 + 위치 인코딩 + MLP(BatchNorm/skip)** | 없음 | 수십만~수백만 | Dacon 1위 1단계가 이 구조로 잘 작동함을 실증. 학습·추론 둘 다 가장 가볍다. **1차 추천 — baseline** |
| **양방향 다층 LSTM/GRU + 자모 단위 vocab** | 없음 | 수백만~1천만대 | Dacon 3위가 실증(4층, embed/hidden 512). MLP보다 인접 문맥을 더 잘 반영 — 문맥 의존이 큰 연음에 유리할 것으로 예상. **2차 추천 — MLP baseline이 부족할 때 승격** |
| **소형 pretrained 한국어 인코더 fine-tune(예: koelectra-base, ~110M)** | 있음(pretrained) | ~110M | Dacon 1위가 2단계 보정에 사용, 성능 상한을 밀어올림. CPU 추론은 위 두 후보보다 무겁지만 "오프라인 로컬"엔 여전히 부합. **스트레치 옵션 — 1·2차로 부족할 때만 투입, 처음부터 이걸로 시작하지 않는다** |
| 대형 LLM 파인튜닝/앙상블(Dacon 2위류) | 있음 | 1B~24B | "로컬 CPU 오프라인" 제약과 불일치. **기각** |
| Seq2seq encoder-decoder + beam search | 있음/없음 무관 | - | 태스크가 길이 보존 분류 문제로 확인됐으므로 이 복잡도가 불필요. **기각(과설계)** |

### 권장 진행 순서

1. **MLP baseline**부터 구현 — 가장 빠르게 "이 방향이 통하는지" 신호를 얻는다(Dacon 3위처럼 train
   loss 곡선만으로도 1차 판단 가능).
2. 부족하면 **BiLSTM/BiGRU**로 승격(문맥 창을 넓힘).
3. 그래도 부족한 유형(예상: 연음처럼 문맥 의존이 큰 케이스)에 한해 **소형 pretrained 인코더
   fine-tune**을 추가 검토.
4. 대형 LLM 경로는 이번 설계에서 채택하지 않는다.

이 순서 자체가 "가장 단순한 것부터 검증하고, 필요할 때만 복잡도를 올린다"는 실행 가능성 우선
전략이다 — 2인 팀·대회 일정에 맞는다.

### 비용 문제가 구조적으로 줄어드는 이유

`CHOSUNG_GUARDRAIL_IMPACT.md`가 지적한 "최악 16회 분류 필요" 문제는 사전 lookup이 여러 후보를
동시에 내놓는 조합적 구조 때문이었다. 위치별 분류 모델은 애초에 **후보 하나(가장 높은 확률의
조합)를 결정론적으로 출력**하므로 이 문제가 설계 단계에서 상당 부분 해소된다. 슬롯별 확률이
낮아 불확실한 경우에 한해 상위 2~3개 조합만 추가로 내놓는 정도로 제한하면 된다(전부 다시
combinatorial하게 열거하지 않는다).

---

## 데이터 수집 계획

### 합성 페어는 라벨링 비용이 거의 0

`ko_obfuscator.py`의 각 변환은 `(text, intensity, seed) -> text` 결정론적 함수라 정상 문장에
적용하기만 하면 `(obfuscated, original)` 학습 페어가 자동 생성된다.

### 선행 작업: `ko_obfuscator.py` 확장

1. **연음(liaison)·구개음화(palatalize)**: 원격 `experiment/disparity` 브랜치(미병합, 존재 여부
   재확인 필요)에 초기 구현이 있다고 조사됐다. 겹받침(ㄳ·ㄺ·ㄵ 등) 처리를 보강해야 한다.
2. **종성 크래밍**: 현재 구현이 전혀 없다. 신규 구현 필요(무의미한 받침 삽입/치환).
3. 기존 `TRANSFORMS`/`FAMILY` 레지스트리 패턴과 시드 관행을 그대로 따른다.

### 말뭉치 원천과 규모

- **학습용 base 문장 풀**: 위키백과 한국어 문장 샘플(라이선스는 팀이 명시적으로 판단 필요) +
  기존 `hf_repo/seeds/*.jsonl`(CC-BY-4.0, 이미 승인됨). 규모 제안: base 문장 1만~1.5만 개 —
  위치별 자모 분류는 문장이 아니라 슬롯 단위로 학습 신호가 쪼개지므로 실제 학습 예시 수는 문장
  수보다 훨씬 많다(문장 1만 개 × 평균 음절 수 × 3슬롯 = 수백만 슬롯 단위 예시, Dacon 3위가 이
  규모 이하로도 수렴한 것과 일치).
- **dev**: 기존 505 seed(attack 301/benign 204) 재사용, threshold 튜닝·오류 분석용.
- **locked-test**: 독립 신규 80~120 seed. 기존 tensify locked-test 절차 재사용.
- **참고 신호(사용 여부 별도 판단)**: Dacon `train.csv` — 실제 사람이 작성한 한국어 텍스트에 걸린
  음운 노이즈 11,263쌍. k-safeguard 용어의 정확히 같은 카테고리는 아니지만 겹치는 부분(종성
  삽입, 자음 삼중 혼동)이 있어 "이 정도 규모·다양성이면 가벼운 모델도 통한다"는 사전 검증용
  smoke corpus로는 라이선스 확인 없이도 로컬 실험 단계에서 참고할 수 있다(단, 프로젝트 학습
  데이터로 채택하거나 결과를 대외 공개하려면 라이선스 확인 필수).

### 분할 원칙

`seed_id` 기준으로 train(base 문장 풀)·dev(505 seed)·locked-test(신규 80~120 seed)를 완전
분리한다. 505 seed는 학습에 포함하지 않는다.

---

## 게이팅/라우팅 전략 (비용·지연 조절)

### 남은 문제: "이 입력에 모델을 돌릴 필요가 있는가"

분류 방식 자체가 "후보 몇 개를 보낼까"라는 조합 폭발 문제는 줄여주지만, "애초에 이 문장/스팬에
복원 모델을 실행할지"는 여전히 결정해야 한다. 초성체는 호환 자모 존재로, 된소리는 경음 밀도로
저비용 신호를 이미 쓰고 있다(`_chosung_spans`, tensify의 `_tense_positions`). 연음·종성크래밍의
결과물은 문법적으로 정상인 한글 음절이라 이런 값싼 신호가 없다.

### 게이트 설계

- 문자 단위 n-gram 통계 언어모델(경량, GPU 불필요)을 base 문장 풀만으로 학습.
- 어절 ± 인접 1어절 단위 perplexity로 "부자연스러운" 스팬만 무거운 모델로 보낸다.
- threshold는 benign dev(오탐)와 attack dev(미탐)로 스윕해 기본 연결 전에 확정.
- 여러 스팬이 걸리면 배치 순전파로 묶는다.

| 기법 | 1차 필터 | 분류 모델 호출 대상 |
|---|---|---|
| chosung | 호환 자모 + `min_initials>=3`(기존) | 사전 매칭 실패(OOV) 스팬만 |
| tensify | 경음 밀도(기존 파라미터) | 해당 스팬(위치별 분류이므로 소량) |
| liaison/종성크래밍 | n-gram perplexity 게이트(신규) | 게이트 통과 상위 N개 어절만 |

---

## 테스트 방법

### 빠른 proxy 지표 (개발 중, 값싸게 반복)

Dacon 3위의 train loss 곡선처럼, 무거운 가드레일 실험 전에 모델 자체의 복원 정확도를 먼저
확인한다: 슬롯 단위 accuracy(초성/중성/종성 각각), 문자열 exact-match rate, 문자 단위 edit
distance. 이 단계에서 통하지 않으면 가드레일 실험으로 넘어가지 않는다 — 비용을 아낀다.

### 가드레일 영향 평가 (기존 프로토콜 재사용)

`dev_note/EVALUATION_SPEC.md`의 트랙·조건·지표·통계 방법을 그대로 재사용한다.

- 지표: NRR, Residual CER, Recovery Gain, ΔFPR-clean/obf, Exact Restoration, chrF++/BERTScore.
- 통계: seed-cluster bootstrap 10,000회, McNemar+Holm 보정.
- locked-test 성공 기준(1차 채택): NRR 점추정 ≥50%(CI 하한 >25%), ΔFPR-clean/obf 각각 점추정
  ≤2%p·CI 상한 ≤5%p, clean benign Mutation Rate ≤1%, 예외/빈출력/Unicode 손상 0건.
- 추가 지표: Candidate-cap 도달률(위치별 분류 도입 전/후 비교 — 구조적으로 줄어드는지 실측),
  p50/p95/p99 추가 지연시간.

### 실행 단계

1. **Smoke** — 3~5개 시드로 코드 경로만 검증.
2. **Dev/Pilot** — 505 seed로 게이트 threshold, MLP vs BiLSTM 비교, 4개 기법 독립 게이트(부분
   승격 허용).
3. **Locked test** — 신규 80~120 seed로 단일 실행, seal → 단일 실행 → threshold 불변경.

### 해석 제한

기존 `CHOSUNG_GUARDRAIL_IMPACT.md`/`TENSIFY_LOCKED_RESULT.md` 패턴 그대로: 단일 가드레일
결과, 하위 LLM intent comprehension 미측정, 기법 간 조합 미포함, `results/`·`predictions.jsonl`
커밋 제외.

---

## 서빙 계획

### 패키지 구조

```
src/k_safeguard/providers/ml_restore.py   (신규)
```

- 새 extra: `k-safeguard[ml-restore]` — MLP/BiLSTM baseline은 `onnxruntime`만으로 충분(수십MB
  이하). 소형 pretrained 인코더까지 승격할 경우에만 추가 의존성 검토.
- `providers/wordfreq.py`의 optional-dependency 관행(`try/except ImportError`)을 그대로 따른다.
- 자모 인코딩/디코딩은 `chosung.py`의 기존 상수를 재사용해 `tokenizers`/`sentencepiece` 없이
  순수 Python으로 구현.

### 지연시간 예산 (제안, 사전등록 대상)

위치별 분류(MLP/BiLSTM)는 beam search가 없는 단일 순전파이므로 이전 seq2seq안보다 예산을 낮출
수 있다.

- MLP baseline: p95 추가 지연 ≤ 100ms.
- BiLSTM 승격: p95 추가 지연 ≤ 200ms.
- 소형 pretrained 인코더 스트레치 옵션 투입 시: p95 추가 지연 ≤ 400ms.

이 값은 제안이며 dev 단계 실측 후 확정한다.

### 모델 아티팩트 배포

Git에 체크포인트를 넣지 않는다(기존 관행). HF Hub에 revision 고정해 올리고
`ml_restore.py`가 다운로드 경로/캐시를 참조한다.

---

## 새 `CandidateProvider` 구현체 인터페이스 스케치 (코드 없음, 구조만)

- **`SlotClassifier`** (내부 유틸): 자모 임베딩 + (MLP 또는 BiLSTM) + 위치별 68-class softmax.
  `predict(text) -> (restored_text, slot_confidences)`. 알려진 슬롯(예: chosung의 초성)은 입력
  feature로 고정하고 모르는 슬롯만 예측.
- **`MlRestoreProvider`** (`CandidateProvider` 구현): 게이트(§게이팅)를 통과한 스팬만
  `SlotClassifier.predict()`에 넘기고, 평균 슬롯 confidence를 `CandidateProposal.confidence`로
  채운다. 불확실도가 높은 슬롯이 있으면 상위 2~3개 조합만 추가 후보로 낸다(전부 조합 열거 안 함).

### 지켜야 할 기존 계약 (검증 완료)

- `name: str` 필수, `generate(text)`의 `text`는 무손실 정규화 후 문자열(`gateway.py` 122행 확인)
  — 별도 정규화 재적용 불필요.
- `CandidateProposal.confidence`는 0~1(Gateway가 검증, `gateway.py` 128행 확인) — 이 provider가
  프로젝트에서 처음으로 confidence를 실질적으로 채우는 provider가 된다(기존 chosung/tensify는
  전부 `None`).
- provider 내부 예외는 Gateway의 기본 정책(`provider_errors` 흡수, `strict_providers=True`일 때만
  raise)에 맡긴다.
- 어떤 메서드도 block/allow를 반환하지 않는다.

---

## 리스크와 롤백 조건

| 리스크 | 감지 방법 | 조치 |
|---|---|---|
| MLP/BiLSTM이 오탐 증가 | locked-test ΔFPR-obf 초과 | 해당 기법만 기본 비활성, `DO_NOT_PROMOTE` 라벨 |
| 슬롯 분류가 원문과 무관한 조합 생성(문맥 무시) | Exact Restoration/chrF++ 급락 | BiLSTM/pretrained 인코더로 승격 검토 |
| 지연시간 예산 초과 | p95 실측 | 더 가벼운 아키텍처로 강등, 그래도 초과 시 opt-in 유지 |
| perplexity 게이트 미탐 과다(연음/종성크래밍) | 오라클 대비 회복률 저하 | threshold 완화 또는 게이트 없이 전수 검사 |
| Dacon 데이터 라이선스 미해결 | 팀 판단 필요 | 위키/505 seed만으로 축소 진행 |

모든 조치는 "기본 비활성, opt-in만 유지" — 패키지의 "기본 Gateway 경로는 무손실 정규화만 연결"
불변식을 어떤 경우에도 건드리지 않는다.

---

## 단계적 로드맵

- **Phase 1(최고 ROI)**: 종성크래밍 신규 구현 + liaison/palatalize 포팅·보강, MLP baseline으로
  chosung/tensify 위치별 분류 학습·평가.
- **Phase 2**: BiLSTM 승격(필요 시), liaison/종성크래밍 생성 + perplexity 게이트 도입·튜닝.
- **Phase 3(스트레치)**: 소형 pretrained 인코더 fine-tune 검토(부족한 유형에 한해).

각 Phase는 dev→locked-test 게이트를 독립적으로 통과해야 다음 Phase 승격 자격을 얻는다.

---

## 브랜치 계획

기존 `AGENTS.md` Git 워크플로 컨벤션(`feature/<산출물명>`, `experiment/<실험명>`, `main`은 항상
데모 가능 상태 유지, 실험은 결과 요약과 함께 PR)을 그대로 따른다. 실험 브랜치는 로드맵의 Phase와
1:1로 매핑하지 않고 "무엇을 검증하느냐" 단위로 쪼갠다 — 하나가 막혀도 나머지가 계속 진행되게 하기
위해서다.

```
main (항상 데모 가능한 상태 유지)
│
├── feature/obfuscator-liaison-crammed
│     ko_obfuscator.py에 liaison·palatalize·jongseong-cramming 생성기 추가 + 경계 케이스
│     단위테스트(빈 입력, 문장부호, 겹받침, 한영 혼용). 모델과 무관하게 그 자체로 유용한 독립
│     레드팀 라이브러리 확장이라 가장 먼저, 독자적으로 PR·merge 가능.
│     └─▶ main
│
├── experiment/ml-restore-corpus            (위 브랜치 merge 후 분기)
│     학습용 base 문장 풀 확보(위키 라이선스 판단 포함) + 신규 생성기로 합성
│     (obfuscated, original) 페어 생성 + train/dev/locked-test seed_id 분할 로직.
│     대용량 코퍼스 파일 자체는 Git에 넣지 않고 생성 코드·분할 로직만 PR.
│     └─▶ main
│
├── experiment/ml-restore-mlp-baseline       (corpus 브랜치 merge 후 분기)
│     자모 슬롯 MLP 분류기 학습 + proxy 지표(슬롯 accuracy/exact-match/edit distance) + dev 평가.
│     ├─ locked-test 통과 → main에 학습 코드 + 결과 문서(TENSIFY_LOCKED_RESULT.md 형식) PR,
│     │                     opt-in provider 상태로 merge(자동 활성화 아님)
│     └─ 미달 → experiment/ml-restore-bilstm 로 이어서 분기(같은 corpus 재사용)
│
├── experiment/ml-restore-bilstm             (MLP 미달 시에만 분기)
│     BiLSTM/BiGRU 승격, 동일 평가 프로토콜. 통과/미달 처리는 위와 동일.
│
├── experiment/ml-restore-gate               (corpus 브랜치 merge 후 분기, 모델 실험과 병행 가능)
│     연음·종성크래밍용 n-gram perplexity 게이트 학습 + threshold 스윕(benign 오탐 vs attack 미탐).
│     └─▶ main (결과 문서 포함)
│
├── experiment/ml-restore-pretrained-encoder (Phase 3 스트레치, MLP·BiLSTM 둘 다 부족한 유형에 한해서만 분기)
│     소형 pretrained 한국어 인코더(예: koelectra-base류) fine-tune 시도.
│     └─▶ main (결과 문서 포함)
│
└── feature/ml-restore-provider              (아래 조건 충족 후 분기)
      착수 조건: 위 experiment 브랜치 중 모델 계열 최소 1개 + ml-restore-gate가 각자 locked-test를
      거쳐(통과든 DO_NOT_PROMOTE든) main에 merge된 뒤.
      src/k_safeguard/providers/ml_restore.py의 SlotClassifier/MlRestoreProvider 실제 구현,
      pyproject.toml [ml-restore] extra 추가, HF Hub 모델 아티팩트 연결.
      └─▶ main (opt-in 상태로 병합 — 등록하지 않으면 기본 Gateway 동작은 그대로)
```

### 병합 순서 요약

1. `feature/obfuscator-liaison-crammed` — 독립적으로 가장 먼저.
2. `experiment/ml-restore-corpus` — 1 완료 후.
3. `experiment/ml-restore-mlp-baseline`(필요 시 `-bilstm`, `-pretrained-encoder`)와
   `experiment/ml-restore-gate` — 2 완료 후, 서로 병렬 진행 가능.
4. `feature/ml-restore-provider` — 3의 모델 계열 중 최소 1개와 게이트 실험이 각각 merge된 뒤 착수.

실험 브랜치가 locked-test에서 성공 기준에 못 미쳐도(§리스크와 롤백 조건) 코드와 결과 문서는
`DO_NOT_PROMOTE` 라벨로 merge한다 — 기존 `TENSIFY_LOCKED_RESULT.md` 선례와 동일하게, "실패도
기록해 재작업을 막는다"는 원칙을 따른다. 실제 활성화 여부만 opt-in으로 갈린다.

이 브랜치 그림은 지금 시점의 제안이며, 착수 시점에 저장소 상태(특히 이 세션 중 확인된 동시 세션의
브랜치 변경)를 다시 확인한 뒤 확정한다.

---

## 검증 방법 (실제 착수 시점 기준 — 지금은 실행하지 않음)

1. Phase 1 코드 착수 시: `python -m unittest discover -s tests`로 기존 테스트 회귀가 없는지 먼저
   확인한 뒤, 신규 `ko_obfuscator.py` 생성기·`ml_restore.py` provider에 경계 케이스 테스트(빈
   입력, 문장부호, 한영 혼용, 독립 자모, 겹받침) 추가.
2. 모델 학습 전 빠른 proxy 지표(슬롯 accuracy, exact-match, edit distance)로 1차 신호 확인 —
   여기서 통하지 않으면 가드레일 실험으로 넘어가지 않는다.
3. 모델 학습/추론 실험은 `experiments/guardrail/enter-env.ps1`로 격리 환경을 활성화한 뒤 실행 —
   GPU와 가중치가 없는 환경에서는 실행했다고 보고하지 말고 미검증 범위를 명시.
4. 각 Phase의 locked-test는 사전등록 문서를 먼저 커밋(seal)한 뒤 단일 실행 — 기존
   `TENSIFY_LOCKED_RESULT.md`/`CHOSUNG_GUARDRAIL_IMPACT.md` 형식을 그대로 따른다.

### 착수 전 재확인이 필요한 참조

이 세션 도중 동시 실행 중이던 다른 Claude Code 세션이 같은 작업 디렉터리에서 여러 브랜치를
병합하며 저장소 상태를 계속 바꿨다. 구현 착수 전 `AGENTS.md` 현재 상태, `experiment/disparity`
브랜치의 실재 여부와 liaison/palatalize 구현 내용, `dev_note/EVALUATION_SPEC.md` 최신 버전을
다시 확인할 것.

---

## 이 문서의 처리 방법

이번 요청은 견적/설계 검토이며 k-safeguard 저장소에는 어떤 파일도 만들거나 고치지 않았다 — 브랜치
생성, 커밋, PR도 하지 않았다. 처리할 미해결 사안(SLM 관련 결정 로그 상태, 동시 세션으로 인한 저장소
불안정, Dacon 데이터 라이선스, 위키 코퍼스 라이선스)이 남아 있어 실제 작업 디렉터리에 바로 반영하지
않았다.

이 문서는 그 결정에 따라 `C:\Users\reals\Desktop\내꺼\프로그램, 대회\OSS 2026\ML\` 아래에만
저장했다. k-safeguard 저장소는 건드리지 않았다.
