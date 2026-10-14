# 가드레일 후보

2026-10-13 HF Hub 조사 결과다. **아직 선정하지 않았다** — 후보만 기록해 두고 선정은 나중에
한다(2026-10-14 결정). 선정 기준은 사용자가 제시한 것을 따른다: 축마다 3개 이상, 최신성·
평가 점수·사용량.

- 메타데이터(`dl30`=최근 30일 다운로드, 공개일, 파라미터, 가중치 크기, 접근 제한, 라이선스,
  카드의 한국어 표시)는 2026-10-13에 `huggingface_hub.HfApi.model_info`로 받은 값이다.
- **성능 수치는 전부 제작자가 자기 평가셋으로 보고한 값이다.** 독립 비교 리더보드는 찾지 못했다.
- "한국어"는 모델 카드의 language 표시만 본 것이다. 실제 한국어 성능은 재 봐야 안다.

## 프롬프트 공격(인젝션) 축 — 우선

| 모델 | 공개 | 파라미터 / 가중치 | dl30 | 접근 | 라이선스 | 한국어 | 메모 |
|---|---|---|---|---|---|---|---|
| [kakaocorp/kanana-safeguard-prompt-2.1b](https://hf.co/kakaocorp/kanana-safeguard-prompt-2.1b) | 2025-05 | 2.09B / 4.2GB | 630 | 공개 | Apache-2.0 | ○ | **현재 사용.** 자체 한국어 평가 F1 0.844(정밀도 0.968, 재현율 0.748). 범주 A1 인젝션·A2 프롬프트 유출 |
| [Horizon-Labs/prompt-injection-guard-base](https://hf.co/Horizon-Labs/prompt-injection-guard-base) | 2026-09 | 0.31B / 1.2GB | 1,526 | 공개 | Apache-2.0 | ○ | 가장 최신. 30개 언어 다국어, 한국어 특화 아님. ModernBERT 계열 |
| [Horizon-Labs/prompt-injection-guard-large](https://hf.co/Horizon-Labs/prompt-injection-guard-large) | 2026-09 | 0.57B / 2.3GB | 210 | 공개 | Apache-2.0 | ○ | 위의 큰 판 |
| [vijil/prompt-injection-v5-20260827](https://hf.co/vijil/prompt-injection-v5-20260827) | 2026-08 | 0.31B / 1.2GB | 603 | 수동 승인 | other | ○ | 다국어. 카드 미확인 |
| [meta-llama/Llama-Prompt-Guard-2-86M](https://hf.co/meta-llama/Llama-Prompt-Guard-2-86M) | 2025-04 | 0.28B / 1.1GB | 121,609 | 수동 승인 | other | ✕ | 인젝션 축 사용량 2위 |
| [protectai/deberta-v3-base-prompt-injection-v2](https://hf.co/protectai/deberta-v3-base-prompt-injection-v2) | 2024-04 | 0.18B / 0.7GB | 692,652 | 공개 | Apache-2.0 | ✕ | 사용량 최다. 영어 위주 |
| [qualifire/prompt-injection-sentinel](https://hf.co/qualifire/prompt-injection-sentinel) | 2025-05 | 0.4B / 1.6GB | 2,432 | 자동 승인 | other | ✕ | |
| [FrameByFrame/guardrail-korean-gemma-4-E2B](https://hf.co/FrameByFrame/guardrail-korean-gemma-4-E2B) | 2026-04 | 5.12B / 10.2GB | 28 | 공개 | gemma | ○ | 한국어. 인젝션·유해성·PII 통합. **문자 난독화 내성을 학습했다고 명시** — 단 카드에 나열된 것은 닮은 글자·투명 문자 등 유니코드 수준이고 된소리·연음 같은 자모 수준은 없다. 가드레일이 난독화에 강하면 복원기가 필요 없다는 반론이 되므로 따로 볼 가치가 있다 |
| [kor-prompt-injection-detection/KLUE_roBERTa_based_model](https://hf.co/kor-prompt-injection-detection/KLUE_roBERTa_based_model) | 2026-06 | 0.11B / 0.4GB | 97 | 공개 | Apache-2.0 | ○ | 대학 수업 프로젝트 |
| [prismdata/guardrail-ko-11class](https://hf.co/prismdata/guardrail-ko-11class) | 2026-02 | 0.01B | 696 | 공개 | GPL-3.0 | ○ | GPL이라 탑재 대상으로는 부적합 |
| [ehd0309/ko-guardrail-llm-v1](https://hf.co/ehd0309/ko-guardrail-llm-v1) | 2026-05 | 2B LoRA | 20 | 공개 | CC BY-SA 4.0 | ○ | 개인 연구용. 제작자가 정밀도 0.53이라 단독 사용 부적합하다고 명시 |

## 유해 콘텐츠 축 — 여유가 될 때

| 모델 | 공개 | 파라미터 / 가중치 | dl30 | 접근 | 라이선스 | 한국어 | 메모 |
|---|---|---|---|---|---|---|---|
| [MarkrAI/ksafeguard-8b](https://hf.co/MarkrAI/ksafeguard-8b) | 2026-07 | 8.03B / 16.1GB | 17 | 공개 | Apache-2.0 | ○ | 자체 보고 프롬프트 유해성 F1 0.954(kanana-safeguard-8b 0.804), 정상 오차단율 0.124로 비교 대상 중 최저. **이 머신에서 로드하다 WSL이 재시작됨** |
| [kakaocorp/kanana-safeguard-8b](https://hf.co/kakaocorp/kanana-safeguard-8b) | 2025-05 | 8.03B / 16.1GB | 625 | 공개 | Apache-2.0 | ○ | 카카오. 증오·괴롭힘·성적·범죄·아동 성착취·자살 자해·잘못된 정보 7범주 |
| [nvidia/Nemotron-3.5-Content-Safety](https://hf.co/nvidia/Nemotron-3.5-Content-Safety) | 2026-05 | 4.3B / 8.6GB | 16,264 | 공개 | other | ○ | 다국어, Gemma 3 기반 |
| [Qwen/Qwen3Guard-Gen-4B](https://hf.co/Qwen/Qwen3Guard-Gen-4B) | 2025-09 | 4.41B / 8.8GB | 32,740 | 공개 | Apache-2.0 | 표시 없음 | |
| [Qwen/Qwen3Guard-Gen-0.6B](https://hf.co/Qwen/Qwen3Guard-Gen-0.6B) | 2025-09 | 0.75B / 1.5GB | 187,851 | 공개 | Apache-2.0 | 표시 없음 | 유해성 축에서 이 머신에 올릴 수 있는 크기 |
| [Horizon-Labs/content-safety-guard-base](https://hf.co/Horizon-Labs/content-safety-guard-base) | 2026-09 | 0.31B / 1.2GB | 130 | 공개 | Apache-2.0 | ○ | 최신. 인젝션 쪽과 같은 계열 |
| [iknow-lab/llama-3.2-3B-wildguard-ko](https://hf.co/iknow-lab/llama-3.2-3B-wildguard-ko) | 2024-10 | 3.21B / 6.4GB | 35 | 공개 | llama3.2 | ○ | 한국어 특화. K-SafeGuard 표에서 2위 |
| [google/shieldgemma-2b](https://hf.co/google/shieldgemma-2b) | 2024-07 | 2.61B / 5.2GB | 12,810 | 수동 승인 | gemma | 표시 없음 | |
| [meta-llama/Llama-Guard-3-8B](https://hf.co/meta-llama/Llama-Guard-3-8B) | 2024-07 | 8.03B / 16.1GB | 31,586 | 수동 승인 | llama3.1 | ✕ | |
| [meta-llama/Llama-Guard-4-12B](https://hf.co/meta-llama/Llama-Guard-4-12B) | 2025-04 | 12.0B / 24.0GB | 48,697 | 수동 승인 | other | ✕ | |
| [ibm-granite/granite-guardian-3.3-8b](https://hf.co/ibm-granite/granite-guardian-3.3-8b) | 2025-06 | 8.17B / 16.3GB | 3,696 | 공개 | Apache-2.0 | ✕ | |
| [allenai/wildguard](https://hf.co/allenai/wildguard) | 2024-06 | 7.25B / 29.0GB | 137,365 | 자동 승인 | Apache-2.0 | ✕ | |

제외: [K-intelligence/Llama-SafetyGuard-Content-Binary](https://hf.co/K-intelligence/Llama-SafetyGuard-Content-Binary)(KT) —
프롬프트가 아니라 **모델 응답**을 판별하는 모델이라 용도가 다르다.

## Kanana 선택에 대한 판단 (2026-10-13)

- 인젝션 축에서는 여전히 타당하다. 한국어 특화이고 분류 체계가 문서화된 대형사 공개 모델은
  이것뿐이고, 더 나은 한국어 인젝션 가드레일이 있다는 근거는 찾지 못했다.
- 다만 공개된 지 1년 반이 지났고 더 작고 새로운 다국어 인코더들이 나왔다. 한국어에서 어느
  쪽이 나은지는 직접 재 봐야 안다.
- 제안은 교체가 아니라 **가드레일을 여럿 두는 것**이다.

## 이 머신에서 측정한 속도 (2026-10-13)

`transformers 4.57.6`, fp16, RTX 4050 6GB.

| 모델 | 속도 | VRAM |
|---|---|---|
| Horizon-Labs prompt-injection-guard-base (0.31B) | 1,184행/초 | 0.7GB |
| kanana-safeguard-prompt-2.1b | 57행/초 | 4.5GB |
| MarkrAI/ksafeguard-8b (4비트 로드 시도) | 로드 중 WSL 재시작 | — |

기존 벤치마크(`k-safeguard/hf_repo/benchmark.jsonl`, 8,585행 = 505 시드 × 원문 1 + 기법 8종 ×
강도 2, 문장 길이 중앙값 31자)를 기준으로 인코더는 약 7초, Kanana 프롬프트 모델은 약
2분 30초다. 인젝션 축은 시간이 제약이 아니다. 유해성 축은 후보 대부분이 3B~8B라 이
머신에서는 돌리기 어렵다.
