# 데이터셋 조사와 사용 결정

두 트랙이 함께 쓰는 데이터를 정리한다. 조사는 2026-10-13 HF Hub(텍스트 전용만, 이미지가 섞인
것은 제외), 결정은 2026-10-14다. 내려받은 샘플은 `data/hf-survey/`(커밋 제외)에 있다.

## 결정

- **KOTOX를 쓴다.** 라이선스(MIT)가 확실한 유일한 난독화↔원문 쌍 데이터다. 다른 후보의
  라이선스는 정리되기 전까지 쓰지 않는다.
- **505 dev 시드는 지금처럼 평가 전용**으로 둔다.
- **평범한 대화형 정상 문장은 확보가 어렵다고 본다.** 공개 데이터에서 찾지 못했다. benign
  세트 확대(지금 204개, 오변경 ≤1% 판정에 부족)는 다른 방법이 필요한 미해결 과제로 남긴다.

## 필요한 데이터 세 종류

| 종류 | 용도 | 지금 |
|---|---|---|
| 깨끗한 공격 문장 | 직접 난독화해 학습·평가 쌍을 만드는 재료 | 505 시드의 공격 301개 |
| 사람이 실제로 쓴 난독화 | 규칙으로 만든 난독화가 현실을 반영하는지 검증 | 없음 |
| 대화형 정상 문장 | 정상 문장 오변경 1% 기준 판정 | 505 시드의 정상 204개 |

## KOTOX — 채택

[ssgyejin/KOTOX](https://hf.co/datasets/ssgyejin/KOTOX) · MIT ·
논문 [arXiv 2510.10961](https://arxiv.org/abs/2510.10961) · 코드 [GitHub](https://github.com/leeyejin1231/KOTOX)

- **내용**: 한 행에 정상 문장, 유해 문장, 각각의 난독화본, 적용 규칙 코드가 있다. 메신저·
  댓글체다. 원문은 K/DA에서 사람이 거른 2,294쌍이다.
- **만든 방식**: 실제 리뷰 144건에서 도출한 규칙 17종(음운·시각·음차·통사·화용 5분류)을
  기계로 적용했다. 행마다 규칙 2·3·4개(easy·normal·hard)를 건다. 음차·의미 치환 일부는 LLM으로
  만들었다. **사람이 직접 쓴 난독화는 아니다.**
- **받아 둔 것**: `data/hf-survey/ssgyejin__KOTOX/data/` — `total/train.csv` 5,505행,
  `easy/test.csv` 230행, `hard/test.csv` 230행. 논문 기준 valid 687·test 690 중 나머지는
  받지 않았다.
- **확인된 주의점**
  - **복원 정답이 될 수 없는 쌍이 섞여 있다.** 의미 번역(`비싼→익스펜시브`,
    `지금→나우`), 단어 교체(`정말 좋아→진짜 gㅔ 좋아`), 조사 탈락(`두명이→two명`)이 있다.
    글자 단위 복원기 학습에 그대로 넣으면 없는 내용을 만들어내도록 가르치게 된다.
  - **위치 정렬이 없다.** 라벨이 문장당 규칙 코드 목록뿐이라 어느 글자가 바뀌었는지 모른다.
  - **라벨 코드가 논문 번호와 맞지 않는다.** 데이터의 코드(`1-1`, `1-3`, `2-1`, `5-1`, `6-1`,
    `8-1`~`8-3`, `10`, `11`, `13-2` 등 17종)는 논문의 5분류 번호 체계로 해석되지 않는다.
    KOTOX 코드에서 대응표를 확인해야 한다(미해결).
  - 실제 리뷰 144건(논문의 wild 평가셋)이 공개됐는지는 확인하지 못했다.
- **쓰는 방법(논의 중, 실험 계획 미확정)**: 원본 쌍은 "남의 생성기" 평가 기준으로 쓰고,
  학습에는 원본 난독화문 대신 결정론적 규칙을 자체 생성기로 이식해 쓰는 방향을 검토했다.

## 그 밖의 후보 — 보류

라이선스 확인 전까지 쓰지 않는다. 학습에 쓸 수 있는 라이선스는 MIT·Apache·CC BY 계열을
전제로 정리했다.

### 공격 문장

| 데이터셋 | 규모 | 라이선스 | 메모 |
|---|---|---|---|
| [MarkrAI/ko-jailbreak](https://hf.co/datasets/MarkrAI/ko-jailbreak) | 11,441 (유해 요청 10,303 + 탈옥 템플릿 1,138) | 행마다 출처 라이선스(MIT·Apache-2.0 등), 카드는 other | 영어 벤치마크 9종 기계 번역, 약 83%가 SALAD-Bench. 번역투가 뚜렷하고 유해성이 약한 문장이 섞여 선별 필요 |
| [AIM-Intelligence/XL-SafetyBench](https://hf.co/datasets/AIM-Intelligence/XL-SafetyBench) 한국 | 탈옥 450 + 문화 민감 100 | CC BY 4.0 | 번역이 아니라 한국 맥락으로 작성. 한국어가 가장 자연스럽다. 소규모 |
| [nvidia/Nemotron-Safety-Guard-Dataset-v3](https://hf.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3) 한국어 | test 2,947 (유해 1,641 / 정상 1,306) | CC BY 4.0 | 정상·유해가 같은 문체. 라벨 3분의 2가 사람. 번역이지만 ko-jailbreak보다 자연스럽다 |
| [xxxjjhhh/korean_guardrail_test](https://hf.co/datasets/xxxjjhhh/korean_guardrail_test) | 647 (인젝션 352 / 정상 295) | CC BY 4.0 | 대화체, 505 시드와 가장 비슷. "정상"은 인젝션 기준이라 유해 요청이 정상으로 표시돼 있다 |
| [heretic-org/Multilingual-Harmless-Harmful](https://hf.co/datasets/heretic-org/Multilingual-Harmless-Harmful) 한국어 | 정상 416 / 유해 416 | CC BY-SA 4.0 | 번역된 "~하라" 명령체 |
| [SungJoo/KoTox](https://hf.co/datasets/SungJoo/KoTox) | 39,200 | MIT | **비권장.** 템플릿으로 찍어내 거의 같은 문장이 반복된다. KOTOX와 이름만 비슷한 다른 데이터 |

### 정상 문장

| 데이터셋 | 규모 | 라이선스 | 메모 |
|---|---|---|---|
| [MarkrAI/k-overrefusal](https://hf.co/datasets/MarkrAI/k-overrefusal) | 485 (정상 185 / 유해 300) | other | 유해해 보이는 정상 문장(촌철살인, 살인미소 등). 오변경 판정에 가장 까다로운 종류. 일부 라벨 품질 의심 |

### 사람이 실제로 쓴 난독화

| 데이터셋 | 규모 | 라이선스 | 메모 |
|---|---|---|---|
| [TEAMREBOOTT-AI/Ko-ToxicWebQ](https://hf.co/datasets/TEAMREBOOTT-AI/Ko-ToxicWebQ) | 47,714 | CC BY-NC-SA, 접근 승인 필요 | 카드 설명상 실제 Bing 검색어에 유해성·난독화 전략 표시. 비상업이라 쓴다면 평가 전용. 내용 미확인 |
| [jwengr/DACON-Korean-Review-Obfuscation](https://hf.co/datasets/jwengr/DACON-Korean-Review-Obfuscation) | train 9,008 / dev 1,126 / test 1,126 | 표시 없음 | 원본 대회 약관 미해결로 데이터 파일은 열지 않았다 |

## 외부 데이터로 세면 안 되는 것

[kimchunsik03/KoreanGuardrail](https://hf.co/datasets/kimchunsik03/KoreanGuardrail)은 이 저장소의
`hf_repo/`(k-safeguard에서 옮겨 옴)와 같은 데이터다(정상 시드 204개, 같은 파일 구성과 기법 이름). 이미 쓰고
있는 505 시드이므로 외부 검증 자료가 아니다.
