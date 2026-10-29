"""P01.4 — 평가셋 v1 구축 (p01.4-evalset-v1).

학습 없음. 원문 695(505 시드 + 공개 정상 190)에 축 네 개(단일·복합·외부·대조군)를 걸어 평가셋을 만들고, 문장마다
레시피(단계·시드·바뀐 구간)를 남긴다. 문장은 `data/derived/evalset_v1/`(커밋 제외)에, 결과 폴더에는 집계만 둔다.

판정(노트에 실행 전 고정):
- 무작위 300건을 레시피만으로 다시 만들면 100% 같은 문장.
- 길이 보존 단일 기법에서 diff 구간이 글자별 비교와 일치하는 비율 ≥ 99%.
- 안 바뀐 문장은 그 축에서 뺀다(축별 제외 수를 보고).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import evalset as E
from kslab import external_gen as X
from kslab import obfuscate as O
from kslab import record as R
from kslab import recipe as RC

EXPERIMENT = "P01_obf_generator"
OUT_DIR = R.DATA_DIR / "derived" / "evalset_v1"
REBUILD_SAMPLE = 300
ALIGN_BAR = 0.99


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def header(gen_seed: int) -> dict:
    ext = {name: {"repo": g.repo, "source": g.source, "license": g.license,
                  "commit": _git(X.EXT_DIR / g.repo, "rev-parse", "HEAD")}
           for name, g in X.GENERATORS.items()}
    ext["kotox"]["g2p_commit"] = _git(X.EXT_DIR / "kotox-src" / "G2P", "rev-parse", "HEAD")
    ext["kotox"]["g2p_license"] = "표시 없음(내부 평가 전용)"
    return {
        "name": "evalset_v1", "gen_seed": gen_seed,
        "code_commit": _git(R.ROOT, "rev-parse", "HEAD"),
        "code_dirty": bool(_git(R.ROOT, "status", "--porcelain")),
        "external": ext,
        "kotox_iconic_sha256": _sha256(O.KOTOX_ICONIC),
        "params": {"single_intensity": E.SINGLE_INTENSITY, "compose_intensity": E.COMPOSE_INTENSITY,
                   "sparse": sorted(E.SPARSE), "sparse_intensity": E.SPARSE_INTENSITY,
                   "compose_rounds": E.COMPOSE_ROUNDS,
                   "compose_sizes": list(E.COMPOSE_SIZES), "attackers": list(E.ATTACKERS),
                   "samples": E.SAMPLES, "external_ops": [f"{g}/{o}" for g, o in E.EXTERNAL_OPS],
                   "control_ops": [f"{g}/{o}" for g, o in E.CONTROL_OPS],
                   "compose_order": list(E.COMPOSE_ORDER),
                   "human_weights": E.type_weights("human")},
        "rule": "기법을 걸어도 안 바뀐 문장은 그 축에서 뺀다. 복합은 안 걸린 기법을 다른 유형으로 바꿔 다시 뽑고"
                "(dropped_attempts), 실제로 걸린 기법만 applied_ops에 남긴다.",
    }


def _record(src: E.Source, axis: str, name: str, steps: list[RC.Step], res: RC.Result,
            applied_ops: list[str], nominal: int) -> dict:
    return {
        "id": f"{src.source_id}/{axis}/{name}",
        "source_id": src.source_id, "origin": src.origin, "label": src.label,
        "axis": axis, "name": name, "nominal_size": nominal, "n_applied": len(applied_ops),
        "applied_ops": applied_ops, "recipe": [s.to_dict() for s in steps],
        "clean": src.text, "obfuscated": res.text,
        "edits": [{"step": e.step, "op": e.op, "before_span": e.before_span, "after_span": e.after_span,
                   "orig_span": e.orig_span, "before": e.before, "after": e.after} for e in res.edits],
    }


def build(sources: list[E.Source], gen_seed: int) -> tuple[list[dict], Counter, Counter]:
    records, made, dropped = [], Counter(), Counter()

    def keep(rec: dict | None, key: str) -> None:
        key = key.replace(":", "_")   # MLflow 지표 이름에 콜론을 쓸 수 없다
        if rec is None:
            dropped[key] += 1
        else:
            made[key] += 1
            records.append(rec)

    for src in sources:
        for tech in E.internal_techniques():
            steps = E.single_steps(tech, E.stable_seed(gen_seed, src.source_id, "single", tech))
            res = RC.run_internal(src.text, steps)
            keep(_record(src, "single", tech, steps, res, [tech], 1) if res.applied else None, f"single/{tech}")
        for size in E.COMPOSE_SIZES:
            for attacker in E.ATTACKERS:
                for sample in range(E.SAMPLES):
                    name = f"{attacker}{size}.{sample}"
                    steps, tried = E.compose_for(src.text, size, attacker,
                                                 E.stable_seed(gen_seed, src.source_id, name))
                    res = RC.run_internal(src.text, steps)
                    ops = [steps[i].op for i in res.applied]
                    rec = _record(src, "compose", name, steps, res, ops, size) if ops else None
                    if rec is not None:
                        rec["dropped_attempts"] = tried
                    keep(rec, f"compose/{attacker}{size}")

    for axis, ops in (("external", E.EXTERNAL_OPS), ("control", None)):
        jobs: dict[str, list[tuple[E.Source, str, RC.Step]]] = defaultdict(list)
        for index, src in enumerate(sources):
            chosen = ops if ops else (E.CONTROL_OPS[index % len(E.CONTROL_OPS)],)
            for gen, op in chosen:
                step = RC.Step(gen, op, 0.0, E.stable_seed(gen_seed, src.source_id, axis, gen, op))
                jobs[gen].append((src, op, step))
        for gen, items in jobs.items():
            outs = X.run(gen, [X.Item(src.text, op, step.seed) for src, op, step in items], timeout=3600)
            for (src, op, step), out in zip(items, outs):
                name = f"{gen}/{op}"
                if gen == "kotox" and "steps" in out:
                    res = RC.from_texts(src.text, out["rules"], out["steps"])
                    applied_ops = [out["rules"][i] for i in res.applied]
                else:
                    res = RC.from_texts(src.text, [op], [out["out"]])
                    applied_ops = [op] if res.applied else []
                nominal = int(op.split(":")[1]) if op.startswith("compose:") else 1
                keep(_record(src, axis, name, [step], res, applied_ops, nominal) if applied_ops else None,
                     f"{axis}/{name}")
    return records, made, dropped


def rebuild_check(records: list[dict], rng: random.Random) -> dict:
    sample = rng.sample(records, min(REBUILD_SAMPLE, len(records)))
    external: dict[str, list[dict]] = defaultdict(list)
    same = 0
    for rec in sample:
        steps = [RC.Step(**s) for s in rec["recipe"]]
        if steps[0].generator == RC.INTERNAL:
            same += RC.run_internal(rec["clean"], steps).text == rec["obfuscated"]
        else:
            external[steps[0].generator].append(rec)
    for gen, recs in external.items():
        outs = X.run(gen, [X.Item(r["clean"], r["recipe"][0]["op"], r["recipe"][0]["seed"]) for r in recs],
                     timeout=3600)
        same += sum(o["out"] == r["obfuscated"] for o, r in zip(outs, recs))
    return {"n": len(sample), "same": same, "rate": same / len(sample)}


def alignment_check(records: list[dict]) -> dict:
    """길이 보존 단일 기법: diff가 잡은 원문 위치 == 글자별로 다른 위치."""
    n = ok = 0
    for rec in records:
        if rec["axis"] != "single" or rec["name"] not in O.LENGTH_PRESERVING:
            continue
        n += 1
        truth = {i for i, (a, b) in enumerate(zip(rec["clean"], rec["obfuscated"])) if a != b}
        found = {i for e in rec["edits"] for i in range(*e["orig_span"])}
        ok += truth == found
    return {"n": n, "match": ok, "rate": ok / n if n else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p01.4-evalset-v1")
    parser.add_argument("--gen-seed", type=int, default=2026)
    args = parser.parse_args()

    head = header(args.gen_seed)
    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT, config=head)
    sources = E.load_sources()
    records, made, dropped = build(sources, args.gen_seed)
    rebuild = rebuild_check(records, random.Random(args.gen_seed))
    align = alignment_check(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head |= {"n_sources": len(sources), "n_records": len(records),
             "origin_counts": dict(Counter(s.origin for s in sources)),
             "label_counts": dict(Counter(s.label for s in sources))}
    (OUT_DIR / "header.json").write_text(json.dumps(head, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    axis_counts = Counter(r["axis"] for r in records)
    status = "pass" if rebuild["rate"] == 1.0 and (align["rate"] or 0) >= ALIGN_BAR else "fail"
    metrics = {"n_sources": len(sources), "n_records": len(records), "axis_counts": dict(axis_counts),
               "made": dict(made), "dropped": dict(dropped), "rebuild": rebuild, "alignment": align,
               "compose_sizes": {f"{a}->{b}": n for (a, b), n in sorted(Counter(
                   (r["nominal_size"], r["n_applied"]) for r in records if r["axis"] == "compose").items())},
               "status": status}
    lines = [f"# P01.4 평가셋 v1 — 원문 {len(sources)}, 문장 {len(records)}", "",
             f"- 재생성 {rebuild['same']}/{rebuild['n']}, 구간 일치 {align['match']}/{align['n']}", "",
             "| 축/이름 | 만듦 | 제외(안 바뀜) |", "|---|---|---|"]
    for key in sorted(set(made) | set(dropped)):
        lines.append(f"| {key} | {made[key]} | {dropped[key]} |")
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish({k: v for k, v in metrics.items()},
               headline={"n_records": len(records), "rebuild_rate": rebuild["rate"],
                         "alignment_rate": align["rate"] or 0.0})
    print("\n".join(lines[:4]))
    print(f"축별: {dict(axis_counts)}")
    print(f"제외가 많은 칸: {[(k, v) for k, v in dropped.most_common(8)]}")
    print(f"복합 (명목->실제): {metrics['compose_sizes']}")
    print(f"\n[P01.4] {status}  -> {run.directory}  (문장: {OUT_DIR})")


if __name__ == "__main__":
    main()
