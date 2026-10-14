"""E5a — 난독화 생성기 갭 메우기 검증.

모델과 무관하게 그 자체로 쓸모 있는 레드팀 라이브러리 확장이라 독립적으로 돌린다.
저장소 main의 `hf_repo/ko_obfuscator.py`에는 5종밖에 없고, 이 실험실은 셋을 더했다.

- `liaison` / `palatalize` — `experiment/disparity` 브랜치에서 발췌 이식. 원본은
  `JONG_TO_CHO`에 단자음만 담아 **겹받침을 전부 놓쳤다**. 이 구현은 겹받침을 앞/뒤로
  쪼개 뒤 자음만 이월한다.
- `jongseong_cram` — 신규. 저장소에 구현이 없었다.

재는 것:
1. 길이 보존 — 위치별 슬롯 분류가 성립하는 전제
2. 발동률 — 505 시드에서 실제로 몇 %나 변형이 걸리는가 (기법의 실효성)
3. 겹받침 커버리지 — 원본 대비 무엇이 고쳐졌는가
4. 결정론성과 경계 케이스
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import jamo as J
from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R

INTENSITIES = (0.3, 0.5, 1.0)

# 원본 disparity 구현이 놓쳤던 겹받침 사례. 손으로 만든 회귀 세트.
COMPOUND_FINAL_CASES = {
    "읽어서": "일거서", "값을": "갑슬", "앉아서": "안자서", "삶은": "살믄",
    "닭을": "달글", "없어요": "업서요", "짧아서": "짤바서", "핥아": "할타",
    "옳은": "올흔", "많아": "만하", "넋이": "넉시", "젊은": "절믄",
}
EDGE_CASES = ("", " ", "!!!", "abc 123", "ㅋㅋㅋ", "ㅏㅑㅓ", "한글Abc123!?", "가", "강아지")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gen-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()
    texts = [s.text for s in seeds]

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E5a_generators",
        config={
            "transforms": sorted(O.TRANSFORMS),
            "length_preserving_declared": sorted(O.LENGTH_PRESERVING),
            "intensities": list(INTENSITIES),
            "gen_seed": args.gen_seed,
            "seeds": len(seeds),
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    metrics: dict = {"per_transform": {}, "compound_finals": {}, "edge_cases": {}}

    for name in sorted(O.TRANSFORMS):
        fn = O.TRANSFORMS[name]
        entry = {"family": O.FAMILY[name], "declared_length_preserving": name in O.LENGTH_PRESERVING}
        for intensity in INTENSITIES:
            same_length = 0
            changed = 0
            changed_chars = 0
            total_chars = 0
            for text in texts:
                out = fn(text, intensity, seed=args.gen_seed)
                same_length += len(out) == len(text)
                changed += out != text
                total_chars += len(text)
                if len(out) == len(text):
                    changed_chars += sum(a != b for a, b in zip(text, out))
            entry[f"intensity_{intensity}"] = {
                "same_length_rate": same_length / len(texts),
                "fire_rate": changed / len(texts),
                "changed_char_rate": changed_chars / total_chars,
            }
        # 결정론성
        entry["deterministic"] = all(
            fn(t, 0.7, seed=5) == fn(t, 0.7, seed=5) for t in texts[:50]
        )
        metrics["per_transform"][name] = entry

    # 겹받침 회귀
    passed, failures = 0, []
    for source, expected in COMPOUND_FINAL_CASES.items():
        got = O.liaison(source, 1.0, seed=0)
        if got == expected:
            passed += 1
        else:
            failures.append({"input": source, "expected": expected, "got": got})
    metrics["compound_finals"] = {
        "cases": len(COMPOUND_FINAL_CASES),
        "passed": passed,
        "failures": failures,
        "note": "disparity 원본은 이 12건 전부에서 변환이 발동하지 않았다.",
    }

    # 경계 케이스: 예외 없이 문자열을 돌려주는가
    edge = {}
    for name, fn in sorted(O.TRANSFORMS.items()):
        results = {}
        for text in EDGE_CASES:
            try:
                out = fn(text, 1.0, seed=0)
                results[repr(text)] = {"ok": isinstance(out, str), "output": out}
            except Exception as exc:  # noqa: BLE001
                results[repr(text)] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        edge[name] = results
    metrics["edge_cases"] = edge
    metrics["edge_cases_all_ok"] = all(
        r["ok"] for per in edge.values() for r in per.values()
    )

    run.write_text("report.md", _report(metrics))
    run.finish(metrics)
    R.publish_baseline("e5a_generators_v1.json", {"run_id": args.run_id, **metrics},
                       run_id=args.run_id)

    print(f"[E5a] 완료 -> {run.directory}")
    print(f"  겹받침 회귀 {passed}/{len(COMPOUND_FINAL_CASES)} 통과")
    print(f"  경계 케이스 전부 정상: {metrics['edge_cases_all_ok']}")
    for name, entry in metrics["per_transform"].items():
        full = entry["intensity_1.0"]
        print(
            f"  {name:16s} 길이보존 {full['same_length_rate']:6.1%}  "
            f"발동률 {full['fire_rate']:6.1%}  문자변경 {full['changed_char_rate']:5.1%}"
        )


def _report(metrics: dict) -> str:
    lines = [
        "# E5a — 난독화 생성기 갭 메우기",
        "",
        "> 상태: `PROVISIONAL_DEV_ONLY`",
        ">",
        "> 모델과 무관하게 독립적으로 쓸모 있는 레드팀 라이브러리 확장이다.",
        "",
        "## 무엇을 더했나",
        "",
        "저장소 main의 `hf_repo/ko_obfuscator.py`는 5종(jamo_decompose·chosung·tensify·",
        "break_spacing·zwsp_inject)만 갖고 있다. 여기에 셋을 더했다.",
        "",
        "| 생성기 | 출처 | 이 실험실에서 바꾼 것 |",
        "|---|---|---|",
        "| `liaison` | `experiment/disparity` 발췌 | **겹받침 처리 추가** (원본은 단자음만) |",
        "| `palatalize` | `experiment/disparity` 발췌 | 그대로 |",
        "| `jongseong_cram` | 신규 | 저장소에 구현 없었음 |",
        "",
        "`experiment/disparity`는 main보다 4.8만 줄 뒤처진 구 구조 브랜치라 머지하지 않고",
        "음운 변환 섹션만 발췌했다.",
        "",
        "## 505 시드에서의 동작 (intensity 1.0)",
        "",
        "| 생성기 | family | 길이 보존 | 발동률 | 문자 변경률 | 결정론적 |",
        "|---|---|---|---|---|---|",
    ]
    for name, entry in metrics["per_transform"].items():
        full = entry["intensity_1.0"]
        lines.append(
            f"| `{name}` | {entry['family']} | {full['same_length_rate']:.1%} | "
            f"{full['fire_rate']:.1%} | {full['changed_char_rate']:.1%} | "
            f"{'예' if entry['deterministic'] else '**아니오**'} |"
        )
    lines += [
        "",
        "길이 보존은 위치별 슬롯 분류가 성립하기 위한 전제다. `jamo_decompose`·",
        "`break_spacing`·`zwsp_inject`는 길이를 바꾸므로 이 실험실의 모델 대상이 아니다",
        "(대신 무손실 정규화로 처리되는 계열이다).",
        "",
        "## 겹받침 회귀",
        "",
        f"{metrics['compound_finals']['passed']}/{metrics['compound_finals']['cases']} 통과. "
        f"{metrics['compound_finals']['note']}",
        "",
        "| 입력 | 기대 | 결과 |",
        "|---|---|---|",
    ]
    for source, expected in COMPOUND_FINAL_CASES.items():
        got = O.liaison(source, 1.0, seed=0)
        mark = "" if got == expected else " ❌"
        lines.append(f"| {source} | {expected} | {got}{mark} |")

    lines += [
        "",
        "## 경계 케이스",
        "",
        f"빈 문자열·공백·문장부호·영숫자·독립 자모·한영 혼용 {len(EDGE_CASES)}종에 대해 "
        f"8개 생성기 전부 예외 없이 문자열 반환: "
        f"**{'통과' if metrics['edge_cases_all_ok'] else '실패'}**",
        "",
        "## 관찰",
        "",
        "- `palatalize`는 발동률이 매우 낮다. 받침 ㄷ/ㅌ 뒤에 '이'가 오는 환경이 드물어서인데,",
        "  단독 공격 기법으로는 실효성이 낮다는 뜻이다. 구현은 유지하되 모델 실험 우선순위는 낮다.",
        "- `liaison`은 발동률이 높고 길이를 보존한다. 다만 결과물이 정상 한국어처럼 보여",
        "  값싼 탐지 신호가 없다 — E6 게이트가 필요한 이유다.",
        "- `jongseong_cram`은 문자 변경률이 높아 n-gram 게이트로 잘 갈린다(E6 참고).",
        "",
        "## 해석 제한",
        "",
        "- 발동률은 505 dev 시드 기준이다. 다른 도메인 텍스트에서는 달라진다.",
        "- 겹받침 회귀 12건은 손으로 만든 세트라 통계적 표본이 아니다.",
        "- 생성기가 만드는 표기가 실제 한국어 화자의 난독화 습관과 얼마나 닮았는지는",
        "  이 실험이 평가하지 않는다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
