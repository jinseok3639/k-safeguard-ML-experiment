"""P03 판정 검수: 사람이 채울 시트를 만들고, 채운 시트로 판정자 정확도를 계산한다.

make   결과 파일에서 모델·라벨마다 같은 수를 뽑아 CSV 시트를 만든다(Google Sheets로 열어 채운다)
       사람이 채울 열: human_understood, human_complied(공격만), memo. 값은 1(true)·0(false), 판단 불가는 비워 둔다
score  채운 시트를 읽어 판정자와의 일치율, 혼동표, 사람 기준 보정 비율을 출력한다

사용:
  python guardrail/p03/audit_llm.py make --inputs data/clean.jsonl data/p03_subset.jsonl \
      --results guardrail/results/s02-llm/exaone4-32b-awq.jsonl guardrail/results/s02-llm/kanana15-8b.jsonl --per-cell 25 --out guardrail/results/s02-llm/audit_sheet.csv
  python guardrail/p03/audit_llm.py score --sheet guardrail/results/s02-llm/audit_sheet.csv
"""
import argparse
import collections
import csv
import random

from run_llm import read_inputs

FIELDS = ["no", "id", "model", "label", "axis", "types", "clean", "obfuscated", "response",
          "judge_understood", "judge_complied", "canary_leak", "judge_evidence",
          "human_understood", "human_complied", "memo"]


def b2s(v):
    return "" if v is None else str(int(v))


def make(a):
    meta = {r["id"]: r for r in read_inputs(a.inputs)}
    rng = random.Random(a.seed)
    rows = []
    for path in a.results:
        res = read_inputs([path])
        for label in ("attack", "benign"):
            pool = [r for r in res if meta[r["id"]]["label"] == label]
            if not a.include_clean:
                pool = [r for r in pool if not r["id"].endswith("/clean")]
            rows += rng.sample(pool, min(a.per_cell, len(pool)))
    rng.shuffle(rows)  # 모델·라벨 순서가 판단에 영향을 주지 않게 섞는다
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for n, r in enumerate(rows, 1):
            m = meta[r["id"]]
            w.writerow({"no": n, "id": r["id"], "model": r["model"].split("@")[0], "label": m["label"],
                        "axis": m["axis"], "types": " ".join(m["types"]), "clean": m["clean"],
                        "obfuscated": m["obfuscated"], "response": r["response"],
                        "judge_understood": b2s(r["understood"]), "judge_complied": b2s(r["complied"]),
                        "canary_leak": b2s(r["meta"].get("canary_leak")),
                        "judge_evidence": r["meta"].get("evidence") or "",
                        "human_understood": "", "human_complied": "", "memo": ""})
    print(f"{a.out}: {len(rows)}행")


def score(a):
    with open(a.sheet, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for field in ("understood", "complied"):
        print(f"\n## {field}")
        by_model = collections.defaultdict(collections.Counter)
        for r in rows:
            h, j = r[f"human_{field}"].strip(), r[f"judge_{field}"].strip()
            if h not in ("0", "1") or j not in ("0", "1"):
                continue
            for k in (r["model"], "ALL"):
                by_model[k][(j, h)] += 1
        for k, c in by_model.items():
            n = sum(c.values())
            agree = c[("1", "1")] + c[("0", "0")]
            print(f"{k}: n={n}  일치 {100 * agree / n:.1f}%  판정자 비율 {100 * (c[('1', '1')] + c[('1', '0')]) / n:.1f}%  "
                  f"사람 비율 {100 * (c[('1', '1')] + c[('0', '1')]) / n:.1f}%  "
                  f"(판정1·사람0 {c[('1', '0')]}, 판정0·사람1 {c[('0', '1')]})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--results", nargs="+", required=True)
    m.add_argument("--per-cell", type=int, default=25, help="모델 x 라벨 칸마다 뽑을 행 수")
    m.add_argument("--include-clean", action="store_true", help="원문(/clean) 행도 포함")
    m.add_argument("--seed", type=int, default=2026)
    m.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--sheet", required=True)
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
