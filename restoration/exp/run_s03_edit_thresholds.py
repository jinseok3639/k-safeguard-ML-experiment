"""S03 — 편집 종류별 임계값 (s03.1).

S02.4에서 합집합 모델은 공격 문장의 손상 자리를 이미 거의 정확히 찾았지만, benign 1%를 지키려고
모든 기법에 공통으로 건 임계값 하나가 확신도가 낮은 편집(받침 크래밍)을 전부 잘랐다. 이 러너는
다시 학습하지 않고, 같은 모델에 편집 종류별 임계값을 정해 공통 임계값·균등 배분과 비교한다.

## 규칙 (노트에 실행 전 고정)

- **편집 종류**: 모델 예측이 입력과 다른 슬롯 — 초성만 / 모음만 / 받침만 / 두 슬롯 이상.
  추론 때 알 수 있는 정보만 쓴다(기법 이름은 쓰지 않는다).
- **예산**: benign dev(KOTOX 학습 분할 정상 문장)에서 바뀌는 문장 수 합계. 헤드라인 1%.
- **탐욕 배분**: 모든 종류를 benign dev가 하나도 안 바뀌는 가장 낮은 높이에서 시작한다. 한
  걸음마다 각 종류를 '정상 문장이 하나 더 바뀌기 직전까지' 내려 보고, 공격 dev의 CER 감소가
  (새로 쓴 정상 문장 하나당) 가장 많이 느는 종류만 내린다. 예산을 넘거나 느는 종류가 없으면 멈춘다.
- **공격 dev**: 코퍼스 평가 분할(학습에 안 쓴 20%) + 우리 생성기. 505 시드·KOTOX test는 고르는 데
  쓰지 않는다.

## 구현

자리별 예측과 확신도는 모델에서 한 번만 뽑아 두고(`_sites`), 임계값 조합마다 그 캐시로 결과를
다시 만든다(`_apply`). `_apply`가 `restore_many`와 같은 결과를 내는지 실행 중에 검산한다.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import record as R
from kslab.jamo import join
from kslab.models.bilstm import _input_slots
from restoration.exp import run_e11_generator_dependence as E11

EXPERIMENT = "S03_edit_thresholds"

CLASSES = ("cho", "jung", "jong", "multi")
BUDGETS = (0.005, 0.01, 0.02)
HEADLINE_BUDGET = 0.01
DEV_PER_TECHNIQUE = 500

# 판정 기준 (노트 '왜 돌리나'의 표와 같다)
CRAM_BAR = 0.06                # 단일 모델 +12.08%p의 절반
TENSIFY_DROP_BAR = 0.01
KOTOX_BAR = E11.MIN_CER_REDUCTION
BENIGN_BAR = 0.01
BENIGN_ALARM = 0.015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--pair-budget", type=int, default=240_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(E11.SEEDS))
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


# --- 자리 캐시 -----------------------------------------------------------------

def _edit_class(before: tuple[int, int, int], after: list[int]) -> str:
    changed = [slot for slot in range(3) if before[slot] != after[slot]]
    return CLASSES[changed[0]] if len(changed) == 1 else "multi"


def _sites(model, texts: list[str]) -> list[list[tuple[int, str, float, str]]]:
    """문장마다 '임계값이 허락하면 바뀌는 자리' (위치, 편집 종류, 확신도, 바꿀 글자).

    `BiLstmSlotRestorer.restore_many`와 같은 규칙: 후보 위치, 시퀀스 상한, 불완전 예측 제외.
    예측이 입력과 같은 자리는 임계값과 무관하게 아무 일도 없으므로 뺀다.
    """
    proba = model._forward(texts)
    out = []
    for r, text in enumerate(texts):
        row = []
        for c in F.candidate_positions(text, model.technique):
            if c >= model.max_len:
                continue
            before = _input_slots(text[c])
            slots = list(before)
            conf = 1.0
            for slot in model.unknown_slots:
                probs = proba[slot][r, c]
                best = int(probs.argmax())
                slots[slot] = best
                conf *= float(probs[best])
            if min(slots) < 0:
                continue
            char = join(*slots)
            if char != text[c]:
                row.append((c, _edit_class(before, slots), conf, char))
        out.append(row)
    return out


def _apply(texts: list[str], sites, thresholds: dict[str, float]) -> list[str]:
    out = []
    for text, row in zip(texts, sites):
        chars = list(text)
        for c, cls, conf, char in row:
            if conf >= thresholds[cls]:
                chars[c] = char
        out.append("".join(chars))
    return out


def _uniform(value: float) -> dict[str, float]:
    return {cls: value for cls in CLASSES}


# --- 임계값 배분 ---------------------------------------------------------------

def _class_scores(sites) -> list[dict[str, float]]:
    """benign 문장마다 종류별 최대 확신도. 그 종류 임계값이 이 값 이하가 되면 문장이 바뀐다."""
    out = []
    for row in sites:
        best: dict[str, float] = {}
        for _, cls, conf, _ in row:
            best[cls] = max(best.get(cls, 0.0), conf)
        out.append(best)
    return out


def _mutated(scores, thresholds) -> set[int]:
    return {i for i, best in enumerate(scores)
            if any(conf >= thresholds[cls] for cls, conf in best.items())}


def _floor_for(scores, cls: str, mutated: set[int]) -> tuple[float, float | None]:
    """종류 cls를 '아직 안 바뀐 정상 문장을 하나도 더 건드리지 않는' 가장 낮은 높이로.

    반환: (그 높이, 그 아래 다음 경계 점수). 경계가 없으면 0까지 내려도 공짜다.
    """
    free = sorted((best[cls] for i, best in enumerate(scores)
                   if i not in mutated and cls in best and best[cls] > 0.0), reverse=True)
    if not free:
        return 0.0, None
    return math.nextafter(free[0], math.inf), free[0]


def allocate_greedy(benign_sites, attack_gain, budget: int) -> tuple[dict[str, float], list[dict]]:
    """탐욕 배분. attack_gain(thresholds) -> 공격 dev 평균 CER 감소."""
    scores = _class_scores(benign_sites)
    # 시작: 정상 문장을 하나도 안 바꾸는 가장 낮은 높이.
    thresholds = {cls: _floor_for(scores, cls, set())[0] for cls in CLASSES}
    mutated = _mutated(scores, thresholds)
    assert not mutated
    current = attack_gain(thresholds)
    trace = [{"step": 0, "thresholds": dict(thresholds), "benign_mutated": 0, "attack_gain": current}]
    while True:
        best_move = None
        for cls in CLASSES:
            _, boundary = _floor_for(scores, cls, mutated)
            if boundary is None:
                continue
            # 경계 점수의 문장(들)을 내주고, 그다음 경계 직전까지 내린다.
            trial = dict(thresholds)
            trial[cls] = boundary
            newly = _mutated(scores, trial)
            trial[cls] = _floor_for(scores, cls, newly)[0]
            newly = _mutated(scores, trial)
            if len(newly) > budget:
                continue
            gain = attack_gain(trial) - current
            cost = len(newly) - len(mutated)
            rate = gain / max(cost, 1)
            if gain > 0 and (best_move is None or rate > best_move[0]):
                best_move = (rate, cls, trial, newly, gain)
        if best_move is None:
            break
        _, cls, thresholds, mutated, gain = best_move
        current += gain
        trace.append({"step": len(trace), "class": cls, "thresholds": dict(thresholds),
                      "benign_mutated": len(mutated), "attack_gain": current})
    return thresholds, trace


def allocate_even(benign_sites, budget: int) -> dict[str, float]:
    """균등 배분: 종류마다 그 종류 혼자서 benign 문장을 budget // 4 개까지만 바꾸게."""
    scores = _class_scores(benign_sites)
    share = budget // len(CLASSES)
    out = {}
    for cls in CLASSES:
        values = sorted((best[cls] for best in scores if cls in best), reverse=True)
        if len(values) <= share or values[share] <= 0.0:
            out[cls] = 0.0
        else:
            out[cls] = math.nextafter(values[share], math.inf)
    return out


# --- 데이터 -------------------------------------------------------------------

def _attack_dev(args) -> tuple[Path, list[dict]]:
    path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    _, held_out = C.split_by_hash(C.corpus_as_items(C.load_corpus(path)), test_fraction=0.2)
    rows = []
    for technique in E11.ALL_TECHNIQUES:
        rng = random.Random(f"{args.gen_seed}:s03-dev:{technique}")
        items = held_out if len(held_out) <= DEV_PER_TECHNIQUE else rng.sample(held_out, DEV_PER_TECHNIQUE)
        for intensity in E11.INTENSITIES:
            made = P.make_pairs(items, [technique], (intensity,), gen_seed=args.gen_seed,
                                include_identity=False)
            rows += [{"technique": technique, "clean": p.clean, "obfuscated": p.obfuscated}
                     for p in made]
    return path, rows


class _AttackGain:
    """공격 dev 평균 CER 감소. 탐욕 배분이 수백 번 부르므로 문장별 CER를 기억해 둔다."""

    def __init__(self, cleans, obfuscated, sites):
        self.cleans, self.obfuscated, self.sites = cleans, obfuscated, sites
        self.before = [M.char_error_rate(c, o) for c, o in zip(cleans, obfuscated)]
        self.memo: list[dict[str, float]] = [{} for _ in cleans]

    def __call__(self, thresholds) -> float:
        total = 0.0
        restored = _apply(self.obfuscated, self.sites, thresholds)
        for i, text in enumerate(restored):
            if text == self.obfuscated[i]:
                continue
            memo = self.memo[i]
            if text not in memo:
                memo[text] = M.char_error_rate(self.cleans[i], text)
            total += self.before[i] - memo[text]
        return total / len(restored)


# --- 평가 ---------------------------------------------------------------------

def _evaluate(thresholds, cache, eval_sets, kotox, benign) -> dict:
    out: dict = {"thresholds": thresholds, "seed_sets": {}, "kotox": {}, "benign": {}}
    for name, data in eval_sets.items():
        restored = _apply(data["obfuscated"], cache["seed"][name], thresholds)
        out["seed_sets"][name] = E11._cer_rows(data["cleans"], data["obfuscated"], restored,
                                               data["clusters"])
    out["per_technique"] = {
        t: statistics.fmean(out["seed_sets"][f"{t}@{i}"]["cer_reduction"] for i in E11.INTENSITIES)
        for t in E11.ALL_TECHNIQUES
    }
    out["mean_13"] = statistics.fmean(out["per_technique"].values())
    obf = [r["obfuscated"] for r in kotox]
    restored = _apply(obf, cache["kotox"], thresholds)
    for sname, keep in (("all", lambda r: True), ("slot_reversible", lambda r: r["slot_reversible"])):
        picked = [i for i, r in enumerate(kotox) if keep(r)]
        out["kotox"][sname] = E11._cer_rows(
            [kotox[i]["clean"] for i in picked], [obf[i] for i in picked],
            [restored[i] for i in picked], [kotox[i]["cluster"] for i in picked])
    for name, texts in benign.items():
        out["benign"][name] = M.clean_mutation_rate(texts, _apply(texts, cache["benign"][name], thresholds))
    return out


def _check_cache(model, texts, sites, thresholds) -> None:
    expected = [r[0] for r in model.restore_many(texts, thresholds)]
    got = _apply(texts, sites, _uniform(thresholds))
    if expected != got:
        bad = sum(a != b for a, b in zip(expected, got))
        raise AssertionError(f"자리 캐시가 restore_many와 다릅니다: {bad}문장")


def run_stage(run, args) -> tuple[dict, dict]:
    corpus_path, train_items = E11._train_items(args)
    run.add_input("corpus", corpus_path)
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(f"seeds/{name}", P.SEED_DIR / name)
    for name in ("benign_dev.jsonl", "benign_test_kotox.jsonl", "kotox_eval_test.jsonl"):
        run.add_input(f"derived/{name}", E11._derived(name))

    benign = E11._benign_sets()
    eval_sets = E11._seed_eval_sets(E11.ALL_TECHNIQUES, args)
    kotox = E11._read_jsonl(E11._derived("kotox_eval_test.jsonl"))
    _, dev_rows = _attack_dev(args)
    dev_clean = [r["clean"] for r in dev_rows]
    dev_obf = [r["obfuscated"] for r in dev_rows]
    n_dev = len(benign["dev"])
    budgets = {f"{b:.3f}": int(math.floor(b * n_dev + 1e-9)) for b in BUDGETS}

    metrics: dict = {
        "data": {"attack_dev_pairs": len(dev_rows), "benign_dev": n_dev, "budget_sentences": budgets,
                 "kotox_test_pairs": len(kotox)},
        "models": {}, "allocations": {}, "evaluations": {},
    }
    for seed in args.seeds:
        key = f"all/seed{seed}"
        print(f"[S03] 모델 {key} (S02 체크포인트) ...", flush=True)
        model, stats = E11._train_universal(train_items, None, seed, args)
        metrics["models"][key] = stats

        cache = {
            "dev_attack": _sites(model, dev_obf),
            "benign": {name: _sites(model, texts) for name, texts in benign.items()},
            "seed": {name: _sites(model, d["obfuscated"]) for name, d in eval_sets.items()},
            "kotox": _sites(model, [r["obfuscated"] for r in kotox]),
        }
        common = E11._thresholds(model, benign["dev"])
        _check_cache(model, benign["dev"], cache["benign"]["dev"], common[f"{HEADLINE_BUDGET:.3f}"])

        attack_gain = _AttackGain(dev_clean, dev_obf, cache["dev_attack"])

        methods: dict[str, dict[str, float]] = {"none": _uniform(0.0)}
        allocations: dict = {}
        for bkey, sentences in budgets.items():
            methods[f"common@{bkey}"] = _uniform(common[bkey])
            methods[f"even@{bkey}"] = allocate_even(cache["benign"]["dev"], sentences)
            greedy, trace = allocate_greedy(cache["benign"]["dev"], attack_gain, sentences)
            methods[f"greedy@{bkey}"] = greedy
            allocations[bkey] = {"greedy_trace": trace, "budget_sentences": sentences}
        for name, thresholds in methods.items():
            allocations.setdefault("attack_dev_gain", {})[name] = attack_gain(thresholds)
        metrics["allocations"][key] = allocations
        metrics["evaluations"][key] = {
            name: _evaluate(thresholds, cache, eval_sets, kotox, benign)
            for name, thresholds in methods.items()
        }
        print(f"[S03] {key} 받침 크래밍 공통 "
              f"{metrics['evaluations'][key][f'common@{HEADLINE_BUDGET:.3f}']['per_technique']['jongseong_cram']:+.4f}"
              f" → 탐욕 {metrics['evaluations'][key][f'greedy@{HEADLINE_BUDGET:.3f}']['per_technique']['jongseong_cram']:+.4f}",
              flush=True)
        run.write_json("metrics.json", metrics)

    _summarize(metrics, args)
    return metrics, _headline(metrics)


def _avg(metrics, args, method, getter) -> float:
    return statistics.fmean(getter(metrics["evaluations"][f"all/seed{s}"][method]) for s in args.seeds)


def _summarize(metrics: dict, args) -> None:
    summary: dict = {}
    methods = list(metrics["evaluations"][f"all/seed{args.seeds[0]}"])
    for method in methods:
        summary[method] = {
            "jongseong_cram": _avg(metrics, args, method, lambda e: e["per_technique"]["jongseong_cram"]),
            "tensify": _avg(metrics, args, method, lambda e: e["per_technique"]["tensify"]),
            "mean_13": _avg(metrics, args, method, lambda e: e["mean_13"]),
            "kotox_reversible": _avg(metrics, args, method,
                                     lambda e: e["kotox"]["slot_reversible"]["cer_reduction"]),
            "kotox_all": _avg(metrics, args, method, lambda e: e["kotox"]["all"]["cer_reduction"]),
            "benign_dev": _avg(metrics, args, method, lambda e: e["benign"]["dev"]["sentence_rate"]),
            "benign_test_pooled": _avg(metrics, args, method,
                                       lambda e: e["benign"]["test_pooled"]["sentence_rate"]),
            "per_technique": {t: _avg(metrics, args, method, lambda e, t=t: e["per_technique"][t])
                              for t in E11.ALL_TECHNIQUES},
        }
    metrics["summary"] = summary
    bkey = f"{HEADLINE_BUDGET:.3f}"
    greedy, common = summary[f"greedy@{bkey}"], summary[f"common@{bkey}"]
    verdicts = {
        "cram_recovers": greedy["jongseong_cram"] >= CRAM_BAR,
        "no_sacrifice": (greedy["mean_13"] >= common["mean_13"]
                         and common["tensify"] - greedy["tensify"] <= TENSIFY_DROP_BAR),
        "kotox_transfers": greedy["kotox_reversible"] >= KOTOX_BAR,
        "benign_holds": greedy["benign_test_pooled"] <= BENIGN_BAR,
        "benign_alarm": greedy["benign_test_pooled"] > BENIGN_ALARM,
    }
    metrics["verdicts"] = verdicts
    if verdicts["benign_alarm"]:
        status = "INCONCLUSIVE_BENIGN_OVERFIT"
    elif not verdicts["cram_recovers"]:
        status = "CALIBRATION_INSUFFICIENT"
    elif not verdicts["no_sacrifice"]:
        status = "RECOVERS_WITH_SACRIFICE"
    else:
        status = "CALIBRATION_WORKS"
    metrics["status"] = status


def _headline(metrics: dict) -> dict:
    bkey = f"{HEADLINE_BUDGET:.3f}"
    s = metrics["summary"]
    g, c, e = s[f"greedy@{bkey}"], s[f"common@{bkey}"], s[f"even@{bkey}"]
    return {
        "cram_greedy": g["jongseong_cram"], "cram_common": c["jongseong_cram"], "cram_even": e["jongseong_cram"],
        "mean13_greedy": g["mean_13"], "mean13_common": c["mean_13"], "mean13_even": e["mean_13"],
        "tensify_greedy": g["tensify"], "tensify_common": c["tensify"],
        "kotox_reversible_greedy": g["kotox_reversible"], "kotox_reversible_common": c["kotox_reversible"],
        "benign_test_greedy": g["benign_test_pooled"], "benign_test_common": c["benign_test_pooled"],
    }


def report(metrics: dict, args) -> str:
    s = metrics["summary"]
    v = metrics["verdicts"]
    bkey = f"{HEADLINE_BUDGET:.3f}"
    pct = lambda x: f"{x * 100:+.2f}%p"
    rate = lambda x: f"{x:.2%}"
    lines = [
        "# S03.1 편집 종류별 임계값 배분", "",
        f"상태 `{metrics['status']}`. 모델: S02 합집합 BiLSTM 시드 {len(args.seeds)}개(재학습 없음). "
        f"예산 기준: benign dev {metrics['data']['benign_dev']:,}문장. 공격 dev "
        f"{metrics['data']['attack_dev_pairs']:,}쌍(코퍼스 평가 분할 + 우리 생성기).", "",
        "## 판정 (예산 1%, 시드 평균)", "",
        "| 질문 | 기준 | 공통 | 균등 | 탐욕 | 판정 |", "|---|---|---|---|---|---|",
    ]
    c, e, g = s[f"common@{bkey}"], s[f"even@{bkey}"], s[f"greedy@{bkey}"]
    lines += [
        f"| 받침 크래밍 회복 | ≥ {pct(CRAM_BAR)} | {pct(c['jongseong_cram'])} | {pct(e['jongseong_cram'])} | "
        f"{pct(g['jongseong_cram'])} | {'통과' if v['cram_recovers'] else '미달'} |",
        f"| 13기법 평균 | ≥ 공통 | {pct(c['mean_13'])} | {pct(e['mean_13'])} | {pct(g['mean_13'])} | "
        f"{'통과' if v['no_sacrifice'] else '미달'} |",
        f"| 된소리 | 하락 ≤ {pct(TENSIFY_DROP_BAR)} | {pct(c['tensify'])} | {pct(e['tensify'])} | "
        f"{pct(g['tensify'])} | |",
        f"| KOTOX 슬롯 가역 쌍 | ≥ {pct(KOTOX_BAR)} | {pct(c['kotox_reversible'])} | {pct(e['kotox_reversible'])} | "
        f"{pct(g['kotox_reversible'])} | {'통과' if v['kotox_transfers'] else '미달'} |",
        f"| benign test 변형률 | ≤ {rate(BENIGN_BAR)} | {rate(c['benign_test_pooled'])} | "
        f"{rate(e['benign_test_pooled'])} | {rate(g['benign_test_pooled'])} | "
        f"{'통과' if v['benign_holds'] else ('경보' if v['benign_alarm'] else '초과')} |",
        "", f"참고: 임계값 없음 — 받침 크래밍 {pct(s['none']['jongseong_cram'])}, 13기법 평균 "
        f"{pct(s['none']['mean_13'])}, KOTOX 슬롯 가역 {pct(s['none']['kotox_reversible'])}, "
        f"benign test {rate(s['none']['benign_test_pooled'])}.", "",
        "## 예산 곡선 (탐욕 배분)", "",
        "| 예산 | 받침 크래밍 | 13기법 평균 | KOTOX 슬롯 가역 | benign dev | benign test |",
        "|---|---|---|---|---|---|",
    ]
    for b in BUDGETS:
        row = s[f"greedy@{b:.3f}"]
        lines.append(f"| {b:.1%} | {pct(row['jongseong_cram'])} | {pct(row['mean_13'])} | "
                     f"{pct(row['kotox_reversible'])} | {rate(row['benign_dev'])} | "
                     f"{rate(row['benign_test_pooled'])} |")
    lines += ["", "## 탐욕 배분 결과 임계값 (예산 1%)", "", "| 시드 | " + " | ".join(CLASSES) + " |",
              "|---" * (len(CLASSES) + 1) + "|"]
    for seed in args.seeds:
        th = metrics["evaluations"][f"all/seed{seed}"][f"greedy@{bkey}"]["thresholds"]
        lines.append(f"| {seed} | " + " | ".join(f"{th[cls]:.6f}" for cls in CLASSES) + " |")
    lines += ["", "## 기법별 CER 감소 (505 시드, 예산 1%)", "", "| 기법 | 공통 | 균등 | 탐욕 | 임계값 없음 |",
              "|---|---|---|---|---|"]
    for t in E11.ALL_TECHNIQUES:
        lines.append(f"| {t} | {pct(c['per_technique'][t])} | {pct(e['per_technique'][t])} | "
                     f"{pct(g['per_technique'][t])} | {pct(s['none']['per_technique'][t])} |")
    lines += ["", "- 공통 임계값 행은 S02.4(`e11d`)의 `actual`과 같아야 한다(재현 확인).",
              "- benign dev는 KOTOX 댓글체 정상 문장이다. 505 시드 benign(대화형 명령문)과 문체가 다르다."]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run = R.Run.start(
        run_id=args.run_id,
        experiment=EXPERIMENT,
        config={
            "corpus_size": args.corpus_size, "wiki_ratio": args.wiki_ratio,
            "pair_budget": args.pair_budget, "epochs": args.epochs, "seeds": list(args.seeds),
            "gen_seed": args.gen_seed, "device": args.device,
            "classes": list(CLASSES), "budgets": list(BUDGETS), "headline_budget": HEADLINE_BUDGET,
            "dev_per_technique": DEV_PER_TECHNIQUE, "intensities": list(E11.INTENSITIES),
            "techniques": list(E11.ALL_TECHNIQUES),
            "bars": {"cram": CRAM_BAR, "tensify_drop": TENSIFY_DROP_BAR, "kotox": KOTOX_BAR,
                     "benign": BENIGN_BAR, "benign_alarm": BENIGN_ALARM},
            "model": "S02 all (union BiLSTM) checkpoints, no retraining",
            "corpus_license": C.LICENSE_NOTE, "kotox_license": "MIT (ssgyejin/KOTOX)",
        },
    )
    metrics, headline = run_stage(run, args)
    run.write_text("report.md", report(metrics, args))
    run.finish(metrics, headline={k: v for k, v in headline.items() if v is not None})
    print(f"\n[S03] 완료 -> {run.directory}  상태 {metrics['status']}")


if __name__ == "__main__":
    main()
