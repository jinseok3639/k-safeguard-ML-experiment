"""`CandidateProvider` 계약을 만족하는 프로토타입.

k-safeguard 저장소를 **읽기 전용으로만** 참조한다. 설치하지 않고 `src`를 sys.path에
얹어서 실제 protocol·dataclass를 그대로 쓴다. 바이트코드도 쓰지 않게 막아둔다 —
저장소 디렉터리에 아무것도 남기지 않는 것이 이 실험실의 규칙이다.

지켜야 할 계약(저장소 `gateway.py`에서 확인):
- `name: str` 속성
- `generate(text) -> Iterable[CandidateProposal]`
- `confidence`는 `None`이거나 0~1 (Gateway가 검증, 벗어나면 ValueError)
- 입력 `text`는 **무손실 정규화를 마친** 문자열이다 — 정규화를 다시 적용하면 안 된다
- 원문을 다시 내보내지 않는다 (Gateway가 이미 views[0]에 넣었다)
- block/allow를 판단하는 메서드를 두지 않는다
- 생성 순서가 곧 우선순위다 (view 예산이 잘리면 뒤쪽이 버려진다)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

KSAFEGUARD_SRC = Path(__file__).resolve().parents[2] / "k-safeguard" / "src"


def load_ksafeguard():
    """저장소 `src`를 sys.path에 얹고 gateway 심볼을 돌려준다. 쓰기 없음."""
    sys.dont_write_bytecode = True      # 저장소에 .pyc 를 남기지 않는다
    path = str(KSAFEGUARD_SRC)
    if not KSAFEGUARD_SRC.exists():
        raise FileNotFoundError(f"k-safeguard src를 찾을 수 없습니다: {KSAFEGUARD_SRC}")
    if path not in sys.path:
        sys.path.insert(0, path)
    from k_safeguard.gateway import CandidateProposal, CandidateProvider, Gateway

    return CandidateProposal, CandidateProvider, Gateway


@dataclass
class MlRestoreProvider:
    """학습된 슬롯 복원기들을 하나의 opt-in provider로 묶는다.

    `models`는 {기법: 복원기}. 각 복원기는 `restore_many(texts) -> [(복원문, [confidence])]`
    를 제공하면 된다(선형·MLP·BiLSTM 전부 이 계약을 지킨다).

    `thresholds`는 기법별 최소 평균 confidence. E3에서 확인했듯 ΔFPR이 구속 조건인
    상황에서는 확신이 낮을 때 **후보를 아예 내지 않는** 편이 낫다. 임계값을 못 넘으면
    그 기법은 그 입력에 대해 침묵한다.
    """

    models: dict
    thresholds: dict = field(default_factory=dict)
    default_threshold: float = 0.9
    name: str = "ml_restore"
    max_candidates_per_technique: int = 1

    # 생성 순서 = 우선순위. 오탐 위험이 구조적으로 없는 기법을 먼저 낸다.
    # 초성체 후보 위치는 호환 자모 낱자뿐이라 정상 문장에는 나타나지 않는다(E4 참고).
    ORDER = ("chosung", "tensify", "liaison", "palatalize", "jongseong_cram")

    def _ordered(self) -> list[str]:
        known = [t for t in self.ORDER if t in self.models]
        return known + sorted(set(self.models) - set(known))

    def generate(self, text: str) -> Iterator:
        CandidateProposal, _, _ = load_ksafeguard()
        if not isinstance(text, str):
            raise TypeError("text는 str이어야 합니다.")
        emitted = 0
        for technique in self._ordered():
            model = self.models[technique]
            restored, confidences = model.restore_many([text])[0]
            if restored == text:
                continue                    # 바뀐 게 없으면 후보가 아니다
            if not confidences:
                continue
            mean_confidence = sum(confidences) / len(confidences)
            threshold = self.thresholds.get(technique, self.default_threshold)
            if mean_confidence < threshold:
                continue                    # abstention — 확신 없으면 침묵
            yield CandidateProposal(
                text=restored,
                lossy=True,
                # Gateway가 0~1을 검증하므로 방어적으로 자른다.
                confidence=min(1.0, max(0.0, mean_confidence)),
                metadata=(
                    ("technique", technique),
                    ("slots_touched", str(len(confidences))),
                    ("threshold", f"{threshold:.3f}"),
                    ("generator", "kslab-ml-restore-0.1.0"),
                ),
            )
            emitted += 1
            if emitted >= self.max_candidates_per_technique * len(self.models):
                return


def contract_report(provider, texts: Iterable[str]) -> dict:
    """provider가 계약을 지키는지 실제 Gateway에 태워 확인한다."""
    CandidateProposal, CandidateProvider, Gateway = load_ksafeguard()

    findings = {
        "isinstance_candidate_provider": isinstance(provider, CandidateProvider),
        "has_name": isinstance(getattr(provider, "name", None), str),
        "has_no_verdict_method": not any(
            hasattr(provider, attr) for attr in ("block", "allow", "decide", "classify")
        ),
        "violations": [],
        "views_seen": 0,
        "candidates_seen": 0,
    }

    gateway = Gateway(providers=[provider], strict_providers=True)
    for text in texts:
        result = gateway.process(text)
        findings["views_seen"] += len(result.views)
        if result.provider_errors:
            findings["violations"].append(
                {"text": text, "provider_errors": list(result.provider_errors)}
            )
        for view in result.views:
            if view.kind != "candidate":
                continue
            findings["candidates_seen"] += 1
            if view.confidence is not None and not 0.0 <= view.confidence <= 1.0:
                findings["violations"].append(
                    {"text": text, "issue": "confidence out of range", "value": view.confidence}
                )
            if view.text == result.original:
                findings["violations"].append(
                    {"text": text, "issue": "원문을 후보로 다시 내보냄"}
                )
            if not view.lossy:
                findings["violations"].append(
                    {"text": text, "issue": "lossy=False 로 표시된 복원 후보"}
                )
    findings["contract_ok"] = (
        findings["isinstance_candidate_provider"]
        and findings["has_name"]
        and findings["has_no_verdict_method"]
        and not findings["violations"]
    )
    return findings
