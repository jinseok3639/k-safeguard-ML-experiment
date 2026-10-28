"""P04.2 — 정규화기 학습 데이터 생성 (p04.2-train-data).

학습 없음. 위키·KLUE 코퍼스 3만 문장에 복원 대상 유형만 섞어 건 쌍을 만들고(`kslab/train_data.py`), 문서 단위로
train/dev를 나눠 `data/derived/p04_train_v1/`에 쓴다. 문장마다 레시피를 남긴다.

판정(노트에 실행 전 고정): 평가셋 원문과 겹침 0, 경계·범위 밖 유형 0, 레시피 재생성 300/300, 대상 9유형이 train에서
각각 3,000쌍 이상, 정답 태그로 표현 못 하는 쌍 ≤ 1%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import edit_tags as E
from kslab import evalset as V
from kslab import record as R
from kslab import train_data as D
from kslab.recipe import Step, run_internal

EXPERIMENT = "P04_normalizer"
CORPUS = C.CORPUS_DIR / "corpus_n30000_wiki50.jsonl"
OUT = R.ROOT / "data" / "derived" / "p04_train_v1"
MIN_PER_TYPE = 3000
MAX_INEXPRESSIBLE = 0.01
REBUILD_SAMPLE = 300


def _norm(text: str) -> str:
    return " ".join(text.split())


def _split(doc: str) -> str:
    digest = hashlib.sha256(f"{D.SPLIT_SALT}:{doc}".encode()).digest()
    return "dev" if int.from_bytes(digest[:4], "big") < int(D.DEV_FRACTION * (1 << 32)) else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p04.2-train-data")
    parser.add_argument("--gen-seed", type=int, default=2027)
    args = parser.parse_args()

    config = {"corpus": CORPUS.name, "gen_seed": args.gen_seed, "sizes": D.SIZES, "variants": D.VARIANTS,
              "identity_share": D.IDENTITY_SHARE, "intensities": D.INTENSITIES, "dev_fraction": D.DEV_FRACTION,
              "target_types": sorted(D.TARGET_TYPES)}
    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT, config=config)
    run.add_input("corpus", CORPUS)

    rows = C.load_corpus(CORPUS)
    eval_texts = {_norm(s.text) for s in V.load_sources()}
    kept = [r for r in rows if _norm(r.text) not in eval_texts]

    pairs, inexpressible = [], 0
    for r in kept:
        for p in D.make_pairs(r.row_id, r.source, r.text, _split(r.source), args.gen_seed):
            if E.oracle_tags(p.obfuscated, p.clean)[1]:
                inexpressible += 1
                continue
            pairs.append(p)

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for p in pairs:
                if p.split == split:
                    f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    header = {"name": "p04_train_v1", "run_id": args.run_id, **config,
              "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
              "files": {s: hashlib.sha256((OUT / f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "dev")}}
    (OUT / "header.json").write_text(json.dumps(header, ensure_ascii=False, indent=2), encoding="utf-8")

    train = [p for p in pairs if p.split == "train"]
    dev = [p for p in pairs if p.split == "dev"]
    type_counts = Counter(t for p in train for t in p.to_dict()["types"])
    op_counts = Counter(op for p in train for op in p.applied_ops)
    size_counts = Counter((p.size, len(p.applied_ops)) for p in train)
    leaked_types = sorted(set(type_counts) - D.TARGET_TYPES)
    train_docs, dev_docs = {p.source for p in train}, {p.source for p in dev}
    eval_overlap = sum(_norm(p.clean) in eval_texts for p in pairs)

    sample = random.Random(0).sample([p for p in pairs if p.recipe], REBUILD_SAMPLE)
    rebuilt = sum(run_internal(p.clean, [Step(**s.to_dict()) for s in p.recipe]).text == p.obfuscated for p in sample)

    total = len(pairs) + inexpressible
    checks = {"eval_overlap_zero": eval_overlap == 0, "no_boundary_types": not leaked_types,
              "rebuild": rebuilt == REBUILD_SAMPLE, "docs_disjoint": not (train_docs & dev_docs),
              "min_per_type": all(type_counts[t] >= MIN_PER_TYPE for t in D.TARGET_TYPES),
              "inexpressible": inexpressible / max(total, 1) <= MAX_INEXPRESSIBLE}
    metrics = {"n_corpus": len(rows), "n_removed_eval_overlap": len(rows) - len(kept),
               "n_train": len(train), "n_dev": len(dev),
               "n_identity_train": sum(not p.recipe for p in train), "n_inexpressible": inexpressible,
               "type_counts": dict(type_counts), "op_counts": dict(op_counts),
               "size_applied": {f"{s}->{a}": n for (s, a), n in sorted(size_counts.items())},
               "rebuilt": rebuilt, "checks": checks, "status": "pass" if all(checks.values()) else "fail"}

    lines = ["# P04.2 학습 데이터 (p04_train_v1)", "",
             f"- 코퍼스 {len(rows)}문장, 평가셋 원문과 겹쳐 뺀 문장 {len(rows) - len(kept)}",
             f"- train {len(train)}쌍(손대지 않은 쌍 {metrics['n_identity_train']}), dev {len(dev)}쌍, "
             f"태그로 표현 못 해 뺀 쌍 {inexpressible}",
             f"- 레시피 재생성 {rebuilt}/{REBUILD_SAMPLE}", "",
             "## 유형별 train 쌍 수", "", "| 유형 | 쌍 |", "|---|---|"]
    lines += [f"| {t} | {type_counts[t]} |" for t in sorted(D.TARGET_TYPES)]
    lines += ["", "## 고른 기법 수 → 실제로 걸린 수 (train)", "", "| 고른 수 → 걸린 수 | 쌍 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in metrics["size_applied"].items()]
    lines += ["", "## 기법별 train 발동", "", "| 기법 | 쌍 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in op_counts.most_common()]
    lines += ["", "## 예시", "", "| 원문 | 난독화 | 기법 |", "|---|---|---|"]
    lines += [f"| {p.clean} | {p.obfuscated} | {', '.join(p.applied_ops)} |" for p in train[:12] if p.recipe]
    lines += ["", f"판정: {metrics['status']} — {checks}"]
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish(metrics, headline={"n_train": len(train), "n_dev": len(dev),
                                  "min_type_count": min(type_counts[t] for t in D.TARGET_TYPES)})
    print("\n".join(lines[: lines.index("## 기법별 train 발동")]))
    print(f"\n[P04.2] {metrics['status']} {checks} -> {run.directory}")


if __name__ == "__main__":
    main()
