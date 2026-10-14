"""E11 — 복원기는 자체 난독화 생성기에 얼마나 묶여 있나 (s02).

지금까지의 학습 데이터와 평가 데이터(505 시드)는 모두 `kslab/obfuscate.py` 하나에서 나왔다.
그래서 보고된 수치는 "우리 생성기를 뒤집는 능력"만 잰다. 이 실험은 새 방법을 시도하지 않고
그 의존도를 측정한다. 네 부분(`--part`)이 한 단계(s02)를 이룬다.

- **a. 데이터 감사** (학습 없음) — KOTOX 규칙 코드 해독, 규칙별 가역성 분류, benign 분할.
- **b. 기법군 제외 학습** — 기법군 하나를 빼고 학습한 합집합 BiLSTM이 그 기법군을 얼마나
  복원하는가. 넣고 학습한 모델 대비 유지율이 생성기 과적합의 크기다.
- **c. 교차 생성기** — 자체 생성기로만 학습한 모델을 KOTOX 원본 쌍(남이 만든 생성기)에 건다.
- **d. 정답 위치 상한** — 복원기에 손상 위치를 알려주면(완벽한 탐지기) 얼마나 고치나.
  탐지기·수정기 분리가 값어치 있는지 미리 가른다.

## 설계에서 조심한 것

- **임계값은 평가셋에서 고르지 않는다.** benign dev(KOTOX 학습 분할의 정상 문장)에서 문장
  변형률 목표(≤1% 등)를 만족하는 가장 낮은 임계값을 정하고, test에는 그 값을 고정해 평가만
  한다. s01(E8b)이 505 시드에서 임계값을 골라 낙관적이었던 것을 고친다.
- **학습 예산 고정.** 기법군 수가 달라도 모델마다 학습 페어 수를 같게 둔다(기법별 문장 수를
  예산 / 기법 수로 나눔). 대신 제외 모델은 남은 기법을 조금 더 많이 본다 — 보고서에 적는다.
- **시드 반복.** 모델 초기화·순서 시드 3개. 데이터 생성 시드는 고정(같은 데이터, 다른 학습).
- **KOTOX는 학습에 넣지 않는다.** 정상 문장은 benign으로, 난독화 쌍은 평가로만 쓴다.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import record as R
from kslab.jamo import CHO_INDEX, is_syllable, join
from kslab.models.bilstm import BiLstmSlotRestorer, _input_slots

EXPERIMENT = "E11_generator_dependence"

# 기법군. chosung은 E4에서 슬롯 복원 부적합으로 판정돼 뺐다.
FAMILIES: dict[str, tuple[str, ...]] = {
    "tensify": ("tensify", "tensify_rule"),
    "liaison": ("liaison", "liaison_correct"),
    "coda": ("jongseong_cram", "coda_neutralize", "cluster_reduce"),
    "assimilation": ("nasalize", "lateralize", "palatalize", "aspirate", "h_deletion"),
    "vowel": ("vowel_confuse",),
}
ALL_TECHNIQUES = tuple(t for family in FAMILIES.values() for t in family)
INTENSITIES = (0.5, 1.0)
SEEDS = (2026, 2027, 2028)

# 단일 기법 모델(part d). E8b와 같은 세 기법.
SINGLE_TECHNIQUES = ("tensify", "liaison", "jongseong_cram")

# dev에서 임계값을 고를 benign 문장 변형률 목표. 헤드라인은 1%.
MUTATION_TARGETS = (0.0, 0.005, 0.01, 0.02, 0.05)
HEADLINE_TARGET = 0.01
MUTATION_BAR = 0.01
MIN_CER_REDUCTION = 0.02
RETENTION_BAR = 0.50       # 계획서 대응 규칙: 50% 미만이면 생성기 확장을 우선

KOTOX_DIR = R.DATA_DIR / "hf-survey" / "ssgyejin__KOTOX" / "data"
KOTOX_FILES = {"train": "total/train.csv", "test_easy": "easy/test.csv", "test_hard": "hard/test.csv"}
DERIVED_DIR = R.DATA_DIR / "derived" / "s02"

# KOTOX 규칙 코드 -> (이름, 분류). 대응은 KOTOX 저장소 augmentation.py 의 MAP 주석에서
# 확인했다(커밋 a01c917, 2026-05-28). 논문의 5분류 번호와는 체계가 다르다.
#   slot       : 자모 단위 변형. 길이 보존 — 우리 슬롯 복원기의 담당 범위
#   char_swap  : 음절 단위 닮은꼴 치환. 길이는 보존되지만 자모 대응이 아니다
#   lexical    : 단어를 다른 표기·언어로 바꿈. 글자 단위로 되돌릴 수 없다 (8-1·8-3은 LLM)
#   structural : 띄어쓰기·순서·기호. 길이가 바뀐다
KOTOX_RULES: dict[str, tuple[str, str]] = {
    "1-1": ("초성 대치", "slot"),
    "1-3": ("모음 대치", "slot"),
    "1-4": ("받침 대치", "slot"),
    "1-5": ("음운 변동 표기(KoG2P)", "slot"),
    "2-1": ("반모음 첨가", "slot"),
    "2-2": ("받침 추가(뒤 초성 따라)", "slot"),
    "2-3": ("초성 추가", "slot"),
    "3-1": ("연음", "slot"),
    "5-2": ("자모 도상 대치(ㄱㄴㄷ·ㅏㅑㅓ)", "slot"),
    "5-1": ("야민정음(음절 닮은꼴)", "char_swap"),
    "6-1": ("90도 회전", "char_swap"),
    "8-1": ("라틴 음차(LLM)", "lexical"),
    "8-2": ("표기 대치(사전)", "lexical"),
    "8-3": ("외국어 음차·의미 치환(LLM)", "lexical"),
    "10": ("띄어쓰기", "structural"),
    "11": ("배열 교란", "structural"),
    "13-2": ("기호 추가", "structural"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--part", choices=("a", "b", "c", "d"), required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--pair-budget", type=int, default=240_000,
                        help="합집합 모델 하나의 학습 페어 수(기법 수와 무관하게 고정)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--families", nargs="*", default=list(FAMILIES),
                        help="part b에서 제외해 볼 기법군")
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


# --- 공통: 데이터 -----------------------------------------------------------

def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def _read_kotox() -> dict[str, list[dict]]:
    out = {}
    for split, rel in KOTOX_FILES.items():
        with (KOTOX_DIR / rel).open(encoding="utf-8") as handle:
            out[split] = list(csv.DictReader(handle))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _derived(name: str) -> Path:
    path = DERIVED_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 먼저 --part a 를 돌리세요.")
    return path


def _train_items(args):
    path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    rows = C.load_corpus(path)
    train, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)
    return path, train


def _union_name(techniques) -> str:
    return F.composite_name(techniques)


def _budget_pairs(train_items, techniques, args) -> list[P.Pair]:
    """기법마다 같은 수의 문장을 뽑아 (identity + 강도 2개) 페어를 만들고 합집합 이름으로 라벨링.

    기법 수와 무관하게 페어 수가 `pair_budget` 근처로 고정된다. 문장 표본은 기법 이름과
    gen_seed로 결정되므로 같은 기법은 어느 모델에서나 같은 문장을 받는다 — 제외 모델과 전체
    모델의 차이는 "그 기법군을 봤는가"와 "다른 기법을 조금 더 봤는가"뿐이다.
    """
    per_technique = max(1, args.pair_budget // (len(techniques) * (1 + len(INTENSITIES))))
    union = _union_name(techniques)
    pairs: list[P.Pair] = []
    for technique in techniques:
        rng = random.Random(f"{args.gen_seed}:{technique}")
        items = train_items if len(train_items) <= per_technique else rng.sample(train_items, per_technique)
        made = P.make_pairs(items, [technique], INTENSITIES, gen_seed=args.gen_seed, include_identity=True)
        pairs += [dataclasses.replace(p, technique=union) for p in made]
    return pairs


def _checkpoint(label: str, seed: int) -> Path:
    return R.RESULTS_DIR / "_checkpoints" / "e11" / f"{label}__seed{seed}.pt"


def _fit(technique, pairs, seed, label, args) -> tuple[BiLstmSlotRestorer, dict]:
    started = time.time()
    model = BiLstmSlotRestorer(
        technique, max_len=96, embed_dim=96, hidden=112, layers=2,
        epochs=args.epochs, batch_size=128, random_state=seed,
        threads=args.threads, device=args.device,
    ).fit(pairs, checkpoint_path=_checkpoint(label, seed), checkpoint_every=0,
          show_progress=not args.no_progress)
    return model, {
        "fit_seconds": round(time.time() - started, 1),
        "pairs": len(pairs),
        "identity_pairs": sum(1 for p in pairs if p.is_identity),
        "loss": list(model._fit_log),
    }


def _seed_eval_sets(techniques, args) -> dict[str, dict]:
    """505 시드에 기법별·강도별 난독화. 평가 전용."""
    seeds = P.load_seeds()
    items = P.seeds_as_items(seeds)
    cleans = [s.text for s in seeds]
    clusters = [s.seed_id for s in seeds]
    out = {}
    for technique in techniques:
        for intensity in INTENSITIES:
            made = P.make_pairs(items, [technique], (intensity,), gen_seed=args.gen_seed,
                                include_identity=False)
            out[f"{technique}@{intensity}"] = {
                "technique": technique, "obfuscated": [p.obfuscated for p in made],
                "cleans": cleans, "clusters": clusters,
            }
    return out


def _benign_sets() -> dict[str, list[str]]:
    seeds = P.load_seeds()
    dev = [r["text"] for r in _read_jsonl(_derived("benign_dev.jsonl"))]
    kotox_test = [r["text"] for r in _read_jsonl(_derived("benign_test_kotox.jsonl"))]
    seed_benign = [s.text for s in seeds if s.label == "benign"]
    return {"dev": dev, "test_kotox": kotox_test, "test_seed": seed_benign,
            "test_pooled": kotox_test + seed_benign}


# --- 공통: 임계값과 평가 ------------------------------------------------------

def _mutation_scores(model, texts) -> list[float]:
    """문장마다 '임계값이 이 값 이하이면 이 문장이 바뀐다'는 점수.

    BiLSTM `restore_many`는 후보 위치(시퀀스 상한 안쪽) 순서대로 confidence를 내고,
    confidence ≥ 임계값인 자리만 바꾼다. 예측(argmax)은 임계값과 무관하므로 임계값 0에서
    실제로 바뀌는 자리들의 confidence 최댓값이 곧 그 문장이 바뀌는 경계다.
    """
    restored = model.restore_many(texts, 0.0)
    scores = []
    for text, (rest, confidences) in zip(texts, restored):
        positions = [c for c in F.candidate_positions(text, model.technique) if c < model.max_len]
        changed = [conf for c, conf in zip(positions, confidences) if rest[c] != text[c]]
        scores.append(max(changed) if changed else 0.0)
    return scores


def _threshold_for(scores: list[float], target: float) -> float:
    """dev 문장 변형률이 target 이하가 되는 가장 낮은 임계값."""
    ordered = sorted(scores, reverse=True)
    allowed = int(math.floor(target * len(ordered) + 1e-9))
    if allowed >= len(ordered):
        return 0.0
    # allowed번째(0부터)로 큰 점수보다 '조금 더 큰' 값. 그 점수 이하의 문장은 모두 안 바뀐다.
    boundary = ordered[allowed]
    if boundary <= 0.0:
        return 0.0
    return math.nextafter(boundary, math.inf)


def _thresholds(model, dev_texts) -> dict[str, float]:
    scores = _mutation_scores(model, dev_texts)
    out = {f"{t:.3f}": _threshold_for(scores, t) for t in MUTATION_TARGETS}
    out["none"] = 0.0
    return out


def _cer_rows(cleans, obfuscated, restored, clusters):
    gains, damage_before, damage_after = [], 0.0, 0.0
    for cluster, clean, obf, rest in zip(clusters, cleans, obfuscated, restored):
        before = M.char_error_rate(clean, obf)
        after = M.char_error_rate(clean, rest)
        gains.append((cluster, before - after))
        damage_before += before
        damage_after += after
    groups: dict[str, list[float]] = {}
    for cluster, value in gains:
        groups.setdefault(cluster, []).append(value)
    balanced = sum(sum(v) / len(v) for v in groups.values()) / max(len(groups), 1)
    return {
        "cer_reduction": balanced,
        # 손상 중 되돌린 몫. 기법마다 발동률이 달라 절대 CER 감소를 기법군끼리 비교하기
        # 어려울 때 쓴다.
        "repaired_share": (damage_before - damage_after) / damage_before if damage_before else 0.0,
        "exact": sum(1 for c, r in zip(cleans, restored) if c == r) / max(len(cleans), 1),
    }


def _evaluate(model, eval_sets, benign, thresholds) -> dict:
    out: dict = {"thresholds": thresholds, "benign": {}, "sets": {}}
    for key, threshold in thresholds.items():
        out["benign"][key] = {
            name: M.clean_mutation_rate(texts, [r[0] for r in model.restore_many(texts, threshold)])
            for name, texts in benign.items() if name != "dev"
        }
        out["benign"][key]["dev"] = M.clean_mutation_rate(
            benign["dev"], [r[0] for r in model.restore_many(benign["dev"], threshold)]
        )
    for name, data in eval_sets.items():
        rows = {}
        for key, threshold in thresholds.items():
            restored = [r[0] for r in model.restore_many(data["obfuscated"], threshold)]
            rows[key] = _cer_rows(data["cleans"], data["obfuscated"], restored, data["clusters"])
        out["sets"][name] = rows
    return out


def _family_summary(evaluation: dict, family: str, key: str) -> dict:
    """기법군 하나의 (기법 × 강도) 평가를 평균한다."""
    names = [f"{t}@{i}" for t in FAMILIES[family] for i in INTENSITIES]
    rows = [evaluation["sets"][n][key] for n in names if n in evaluation["sets"]]
    if not rows:
        return {}
    return {metric: statistics.fmean(r[metric] for r in rows)
            for metric in ("cer_reduction", "repaired_share", "exact")}


def _mean_range(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "n": len(values)}


# --- part a: 데이터 감사 -------------------------------------------------------

def _row_profile(clean: str, obf: str, rules: list[str]) -> dict:
    classes = sorted({KOTOX_RULES.get(r, ("?", "unknown"))[1] for r in rules})
    same_length = len(clean) == len(obf)
    diffs = [i for i, (a, b) in enumerate(zip(clean, obf)) if a != b] if same_length else []
    # 슬롯 가역: 길이가 같고, 다른 자리가 전부 (원문 음절, 난독화 음절 또는 호환 자모)인 경우.
    slot_reversible = same_length and all(
        is_syllable(clean[i]) and (is_syllable(obf[i]) or obf[i] in CHO_INDEX) for i in diffs
    )
    return {
        "classes": classes,
        "same_length": same_length,
        "slot_reversible": slot_reversible,
        "all_slot_rules": all(KOTOX_RULES.get(r, ("", ""))[1] == "slot" for r in rules),
        "cer": M.char_error_rate(clean, obf),
    }


def part_a(run, args) -> tuple[dict, dict]:
    kotox = _read_kotox()
    for split, rel in KOTOX_FILES.items():
        run.add_input(f"kotox/{split}", KOTOX_DIR / rel, rows=len(kotox[split]))

    seen_codes = sorted({c for rows in kotox.values() for r in rows
                         for c in ast.literal_eval(r["obfuscated_labels"])})
    unknown_codes = [c for c in seen_codes if c not in KOTOX_RULES]

    eval_rows: dict[str, list[dict]] = {"train": [], "test": []}
    for split, rows in kotox.items():
        bucket = "train" if split == "train" else "test"
        for index, row in enumerate(rows):
            rules = ast.literal_eval(row["obfuscated_labels"])
            source_id = hashlib.sha256(_nfc(row["neutral"]).encode("utf-8")).hexdigest()[:12]
            for side in ("neutral", "toxic"):
                clean = _nfc(row[side])
                obf = _nfc(row[f"{side}_obfuscated_texts"])
                eval_rows[bucket].append({
                    "id": f"{split}:{index}:{side}", "cluster": source_id, "split": split,
                    "side": side, "clean": clean, "obfuscated": obf, "rules": rules,
                    **_row_profile(clean, obf, rules),
                })

    # benign: KOTOX 정상 문장. dev = 학습 분할, test = 테스트 분할(easy·hard는 같은 원문).
    dev = sorted({_nfc(r["neutral"]) for r in kotox["train"]})
    test = sorted({_nfc(r["neutral"]) for split in ("test_easy", "test_hard") for r in kotox[split]})
    overlap = set(dev) & set(test)
    if overlap:
        raise AssertionError(f"benign dev/test 가 겹칩니다: {len(overlap)}개")
    seed_benign = {s.text for s in P.load_seeds() if s.label == "benign"}

    _write_jsonl(DERIVED_DIR / "benign_dev.jsonl", [{"text": t} for t in dev])
    _write_jsonl(DERIVED_DIR / "benign_test_kotox.jsonl", [{"text": t} for t in test])
    _write_jsonl(DERIVED_DIR / "kotox_eval_train.jsonl", eval_rows["train"])
    _write_jsonl(DERIVED_DIR / "kotox_eval_test.jsonl", eval_rows["test"])
    for name in ("benign_dev.jsonl", "benign_test_kotox.jsonl",
                 "kotox_eval_train.jsonl", "kotox_eval_test.jsonl"):
        run.add_input(f"derived/{name}", DERIVED_DIR / name)

    per_rule: dict[str, dict] = {}
    for code in seen_codes:
        rows = [r for r in eval_rows["train"] + eval_rows["test"] if code in r["rules"]]
        name, klass = KOTOX_RULES.get(code, ("?", "unknown"))
        per_rule[code] = {
            "name": name, "class": klass, "rows": len(rows),
            "same_length_share": statistics.fmean(r["same_length"] for r in rows),
            "slot_reversible_share": statistics.fmean(r["slot_reversible"] for r in rows),
            "mean_cer": statistics.fmean(r["cer"] for r in rows),
        }

    def share(rows, key):
        return statistics.fmean(r[key] for r in rows) if rows else 0.0

    summary = {}
    for bucket, rows in eval_rows.items():
        summary[bucket] = {
            "pairs": len(rows),
            "same_length_share": share(rows, "same_length"),
            "slot_reversible_share": share(rows, "slot_reversible"),
            "all_slot_rules_share": share(rows, "all_slot_rules"),
            "any_lexical_share": share([{"x": "lexical" in r["classes"]} for r in rows], "x"),
            "mean_cer": share(rows, "cer"),
        }

    metrics = {
        "kotox_rule_codes": {c: {"name": KOTOX_RULES[c][0], "class": KOTOX_RULES[c][1]}
                             for c in seen_codes if c in KOTOX_RULES},
        "unknown_rule_codes": unknown_codes,
        "per_rule": per_rule,
        "eval_pairs": summary,
        "benign": {
            "dev_kotox_train_neutral": len(dev),
            "test_kotox_test_neutral": len(test),
            "test_seed_benign": len(seed_benign),
            "test_pooled": len(test) + len(seed_benign),
            "dev_test_overlap": 0,
            "seed_benign_in_kotox": len(seed_benign & (set(dev) | set(test))),
        },
        "status": "DATA_READY" if not unknown_codes else "UNKNOWN_RULE_CODES",
    }
    headline = {
        "benign_dev": len(dev),
        "benign_test_pooled": len(test) + len(seed_benign),
        "kotox_test_slot_reversible_share": summary["test"]["slot_reversible_share"],
        "kotox_test_all_slot_rules_share": summary["test"]["all_slot_rules_share"],
    }
    return metrics, headline


def report_a(metrics: dict) -> str:
    lines = [
        "# E11a — 데이터 감사 (s02)", "",
        f"> 상태: `{metrics['status']}`", "",
        "## KOTOX 규칙 코드", "",
        "대응은 KOTOX 저장소 `augmentation.py`의 `MAP` 주석(커밋 a01c917)에서 확인했다.", "",
        "| 코드 | 규칙 | 분류 | 행 수 | 길이 보존 | 슬롯 가역 | 평균 CER |",
        "|---|---|---|---|---|---|---|",
    ]
    for code, row in sorted(metrics["per_rule"].items(), key=lambda kv: kv[0]):
        lines.append(
            f"| `{code}` | {row['name']} | {row['class']} | {row['rows']:,} | "
            f"{row['same_length_share']:.1%} | {row['slot_reversible_share']:.1%} | {row['mean_cer']:.3f} |"
        )
    lines += [
        "", "행마다 규칙이 2~4개 걸려 있어 '이 규칙이 포함된 행'의 통계다. 규칙 하나의 효과로 읽으면 안 된다.",
        "", "## 평가 쌍", "",
        "| 분할 | 쌍 | 길이 보존 | 슬롯 가역 | 규칙 전부 slot | lexical 포함 | 평균 CER |",
        "|---|---|---|---|---|---|---|",
    ]
    for bucket, row in metrics["eval_pairs"].items():
        lines.append(
            f"| {bucket} | {row['pairs']:,} | {row['same_length_share']:.1%} | "
            f"{row['slot_reversible_share']:.1%} | {row['all_slot_rules_share']:.1%} | "
            f"{row['any_lexical_share']:.1%} | {row['mean_cer']:.3f} |"
        )
    b = metrics["benign"]
    lines += [
        "", "## benign 분할", "",
        f"- dev(임계값 선정): KOTOX 학습 분할 정상 문장 {b['dev_kotox_train_neutral']:,}개",
        f"- test: KOTOX 테스트 분할 정상 문장 {b['test_kotox_test_neutral']:,}개 + 505 시드 benign "
        f"{b['test_seed_benign']}개 = {b['test_pooled']:,}개",
        f"- dev/test 겹침 {b['dev_test_overlap']}개, 시드 benign 중 KOTOX와 겹침 {b['seed_benign_in_kotox']}개",
    ]
    if metrics["unknown_rule_codes"]:
        lines += ["", f"**해독하지 못한 코드**: {metrics['unknown_rule_codes']}"]
    return "\n".join(lines) + "\n"


# --- part b: 기법군 제외 학습 ---------------------------------------------------

def _universal_label(excluded: str | None) -> str:
    return "all" if excluded is None else f"minus_{excluded}"


def _techniques_without(excluded: str | None) -> tuple[str, ...]:
    return tuple(t for f, ts in FAMILIES.items() if f != excluded for t in ts)


def _train_universal(train_items, excluded, seed, args):
    techniques = _techniques_without(excluded)
    union = _union_name(techniques)
    pairs = _budget_pairs(train_items, techniques, args)
    return _fit(union, pairs, seed, _universal_label(excluded), args)


def part_b(run, args) -> tuple[dict, dict]:
    corpus_path, train_items = _train_items(args)
    run.add_input("corpus", corpus_path)
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(f"seeds/{name}", P.SEED_DIR / name)
    for name in ("benign_dev.jsonl", "benign_test_kotox.jsonl"):
        run.add_input(f"derived/{name}", _derived(name))

    benign = _benign_sets()
    eval_sets = _seed_eval_sets(ALL_TECHNIQUES, args)
    metrics: dict = {"families": {f: list(ts) for f, ts in FAMILIES.items()},
                     "models": {}, "evaluations": {}}

    configs = [None] + [f for f in args.families]
    for excluded in configs:
        label = _universal_label(excluded)
        for seed in args.seeds:
            key = f"{label}/seed{seed}"
            print(f"[E11b] 학습 {key} ...", flush=True)
            model, stats = _train_universal(train_items, excluded, seed, args)
            stats["excluded_family"] = excluded
            metrics["models"][key] = stats
            print(f"[E11b] 평가 {key} ({stats['fit_seconds']}s)", flush=True)
            evaluation = _evaluate(model, eval_sets, benign, _thresholds(model, benign["dev"]))
            evaluation["by_family"] = {
                fam: {k: _family_summary(evaluation, fam, k) for k in evaluation["thresholds"]}
                for fam in FAMILIES
            }
            metrics["evaluations"][key] = evaluation
            run.write_json("metrics.json", metrics)

    _summarize_b(metrics, args)
    return metrics, _headline_b(metrics)


def _summarize_b(metrics: dict, args) -> None:
    key = f"{HEADLINE_TARGET:.3f}"
    retention: dict = {}
    for family in args.families:
        per_seed, held, full = [], [], []
        for seed in args.seeds:
            a = metrics["evaluations"].get(f"all/seed{seed}")
            h = metrics["evaluations"].get(f"minus_{family}/seed{seed}")
            if not a or not h:
                continue
            fa = a["by_family"][family][key]["cer_reduction"]
            fh = h["by_family"][family][key]["cer_reduction"]
            full.append(fa)
            held.append(fh)
            per_seed.append(fh / fa if fa > 0 else None)
        retention[family] = {
            "cer_reduction_included": _mean_range(full),
            "cer_reduction_excluded": _mean_range(held),
            "retention": _mean_range(per_seed),
        }
    metrics["retention_at_1pct"] = retention

    benign_check: dict = {}
    for label, evaluation in metrics["evaluations"].items():
        benign_check[label] = {
            name: evaluation["benign"][key][name]["sentence_rate"]
            for name in ("dev", "test_kotox", "test_seed", "test_pooled")
        }
    metrics["benign_at_1pct_dev_threshold"] = benign_check

    means = [r["retention"]["mean"] for r in retention.values() if r["retention"]["mean"] is not None]
    overall = statistics.fmean(means) if means else None
    metrics["mean_retention"] = overall
    if overall is None:
        metrics["status"] = "INCONCLUSIVE"
    else:
        metrics["status"] = "GENERATOR_BOUND" if overall < RETENTION_BAR else "TRANSFERS"


def _headline_b(metrics: dict) -> dict:
    out = {"mean_retention": metrics["mean_retention"]}
    for family, row in metrics["retention_at_1pct"].items():
        out[f"retention/{family}"] = row["retention"]["mean"]
        out[f"cer_included/{family}"] = row["cer_reduction_included"]["mean"]
        out[f"cer_excluded/{family}"] = row["cer_reduction_excluded"]["mean"]
    return out


def report_b(metrics: dict, args) -> str:
    key = f"{HEADLINE_TARGET:.3f}"
    lines = [
        "# E11b — 기법군 제외 학습 (s02)", "",
        f"> 상태: `{metrics['status']}` · 평균 유지율 "
        + (f"{metrics['mean_retention']:.1%}" if metrics["mean_retention"] is not None else "—"), "",
        "기법군 하나를 빼고 학습한 합집합 BiLSTM이 그 기법군을 얼마나 복원하는가. 유지율 =",
        "(빼고 학습한 모델의 CER 감소) / (넣고 학습한 모델의 CER 감소). 두 모델 모두 benign dev",
        "문장 변형률 1%가 되는 임계값을 dev에서 정해 고정했다.", "",
        f"- 학습: 코퍼스 {args.corpus_size:,}문장, 모델당 페어 약 {args.pair_budget:,}개 고정, "
        f"BiLSTM {args.epochs} epoch, 시드 {list(args.seeds)}, 장치 {args.device}",
        "- 평가: 505 시드에 기법별 강도 0.5·1.0 난독화, 기법군 안에서 평균", "",
        "## 유지율 (dev 1% 임계값)", "",
        "| 기법군 | 넣고 학습 CER 감소 | 빼고 학습 CER 감소 | 유지율 (시드 범위) |",
        "|---|---|---|---|",
    ]
    for family, row in metrics["retention_at_1pct"].items():
        inc, exc, ret = row["cer_reduction_included"], row["cer_reduction_excluded"], row["retention"]
        ret_text = (f"{ret['mean']:.1%} ({ret['min']:.1%}~{ret['max']:.1%})"
                    if ret["mean"] is not None else "—")
        lines.append(
            f"| {family} | {inc['mean'] * 100:+.2f}%p | {exc['mean'] * 100:+.2f}%p | {ret_text} |"
            if inc["mean"] is not None and exc["mean"] is not None else f"| {family} | — | — | — |"
        )
    lines += [
        "", "## 기법군 × 모델 CER 감소 (시드 평균, dev 1% 임계값)", "",
        "| 모델 | " + " | ".join(FAMILIES) + " |", "|---" * (len(FAMILIES) + 1) + "|",
    ]
    labels = ["all"] + [f"minus_{f}" for f in args.families]
    for label in labels:
        cells = []
        for family in FAMILIES:
            values = [metrics["evaluations"][f"{label}/seed{s}"]["by_family"][family][key]["cer_reduction"]
                      for s in args.seeds if f"{label}/seed{s}" in metrics["evaluations"]]
            cells.append(f"{statistics.fmean(values) * 100:+.2f}" if values else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "", "## benign 변형률 (dev 1% 임계값을 test에 고정)", "",
        "| 모델 | dev | test KOTOX | test 시드 | test 합 |", "|---|---|---|---|---|",
    ]
    for label, row in metrics["benign_at_1pct_dev_threshold"].items():
        lines.append(f"| {label} | {row['dev']:.2%} | {row['test_kotox']:.2%} | "
                     f"{row['test_seed']:.2%} | {row['test_pooled']:.2%} |")
    lines += [
        "", "## 해석 제한", "",
        "- 학습 예산을 고정했으므로 제외 모델은 남은 기법을 조금 더 많이 본다.",
        "- 기법군 안 기법 중 발동률이 낮은 것(palatalize 등)은 CER 감소를 희석한다. 기법군끼리 절대값을",
        "  비교하지 말고 같은 기법군 안에서 넣고/빼고를 비교한다.",
        "- benign dev는 KOTOX 댓글체 정상 문장이다. 505 시드 benign(대화형 명령문)과 문체가 다르다.",
        "- GPU 학습은 CPU와 비트 단위로 같지 않다(드롭아웃 난수열). 시드 반복이 그 분산을 함께 잡는다.",
    ]
    return "\n".join(lines) + "\n"


# --- part c: 교차 생성기 -------------------------------------------------------

def part_c(run, args) -> tuple[dict, dict]:
    corpus_path, train_items = _train_items(args)
    run.add_input("corpus", corpus_path)
    for name in ("benign_dev.jsonl", "benign_test_kotox.jsonl",
                 "kotox_eval_train.jsonl", "kotox_eval_test.jsonl"):
        run.add_input(f"derived/{name}", _derived(name))
    benign = _benign_sets()
    kotox = {"test": _read_jsonl(_derived("kotox_eval_test.jsonl")),
             "train": _read_jsonl(_derived("kotox_eval_train.jsonl"))}

    subsets = {
        "all": lambda r: True,
        "slot_reversible": lambda r: r["slot_reversible"],
        "all_slot_rules": lambda r: r["all_slot_rules"],
        "any_lexical": lambda r: "lexical" in r["classes"],
        "any_structural": lambda r: "structural" in r["classes"],
    }
    metrics: dict = {"models": {}, "evaluations": {}}
    for seed in args.seeds:
        key = f"all/seed{seed}"
        print(f"[E11c] 모델 {key} (part b 체크포인트를 이어 받음) ...", flush=True)
        model, stats = _train_universal(train_items, None, seed, args)
        metrics["models"][key] = stats
        thresholds = _thresholds(model, benign["dev"])
        evaluation: dict = {"thresholds": thresholds}
        for split, rows in kotox.items():
            obf = [r["obfuscated"] for r in rows]
            by_threshold = {}
            for tkey in ("none", f"{HEADLINE_TARGET:.3f}"):
                restored = [r[0] for r in model.restore_many(obf, thresholds[tkey])]
                result = {}
                for sname, predicate in subsets.items():
                    picked = [i for i, r in enumerate(rows) if predicate(r)]
                    if not picked:
                        continue
                    row = _cer_rows([rows[i]["clean"] for i in picked], [obf[i] for i in picked],
                                    [restored[i] for i in picked], [rows[i]["cluster"] for i in picked])
                    row["pairs"] = len(picked)
                    row["worsened_share"] = statistics.fmean(
                        M.char_error_rate(rows[i]["clean"], restored[i])
                        > M.char_error_rate(rows[i]["clean"], obf[i]) for i in picked)
                    row["improved_share"] = statistics.fmean(
                        M.char_error_rate(rows[i]["clean"], restored[i])
                        < M.char_error_rate(rows[i]["clean"], obf[i]) for i in picked)
                    result[sname] = row
                per_rule = {}
                for code in KOTOX_RULES:
                    picked = [i for i, r in enumerate(rows) if code in r["rules"]]
                    if picked:
                        per_rule[code] = _cer_rows(
                            [rows[i]["clean"] for i in picked], [obf[i] for i in picked],
                            [restored[i] for i in picked], [rows[i]["cluster"] for i in picked])
                        per_rule[code]["pairs"] = len(picked)
                result["per_rule"] = per_rule
                by_threshold[tkey] = result
            evaluation[split] = by_threshold
        metrics["evaluations"][key] = evaluation
        run.write_json("metrics.json", metrics)

    tkey = f"{HEADLINE_TARGET:.3f}"
    summary: dict = {}
    for split in ("test", "train"):
        summary[split] = {}
        for sname in subsets:
            values = [metrics["evaluations"][f"all/seed{s}"][split][tkey].get(sname, {}).get("cer_reduction")
                      for s in args.seeds]
            worse = [metrics["evaluations"][f"all/seed{s}"][split][tkey].get(sname, {}).get("worsened_share")
                     for s in args.seeds]
            summary[split][sname] = {"cer_reduction": _mean_range(values), "worsened_share": _mean_range(worse)}
        summary[split]["per_rule"] = {
            code: _mean_range([metrics["evaluations"][f"all/seed{s}"][split][tkey]["per_rule"]
                               .get(code, {}).get("cer_reduction") for s in args.seeds])
            for code in KOTOX_RULES
        }
    metrics["summary_at_1pct"] = summary
    reversible = summary["test"]["slot_reversible"]["cer_reduction"]["mean"]
    metrics["status"] = (
        "INCONCLUSIVE" if reversible is None
        else "TRANSFERS_ON_REVERSIBLE" if reversible >= MIN_CER_REDUCTION
        else "DOES_NOT_TRANSFER"
    )
    headline = {
        "kotox_test_all_cer_reduction": summary["test"]["all"]["cer_reduction"]["mean"],
        "kotox_test_reversible_cer_reduction": reversible,
        "kotox_test_all_slot_rules_cer_reduction": summary["test"]["all_slot_rules"]["cer_reduction"]["mean"],
        "kotox_test_worsened_share": summary["test"]["all"]["worsened_share"]["mean"],
    }
    return metrics, headline


def report_c(metrics: dict, args) -> str:
    lines = [
        "# E11c — 교차 생성기: 자체 생성기로 학습한 모델을 KOTOX에 (s02)", "",
        f"> 상태: `{metrics['status']}`", "",
        "모델은 E11b의 `all`(5개 기법군 전부로 학습한 합집합 BiLSTM)과 같다. 임계값은 benign dev",
        "1% 지점. KOTOX 쌍은 학습에 넣지 않았다.", "",
    ]
    for split in ("test", "train"):
        s = metrics["summary_at_1pct"][split]
        lines += [
            f"## KOTOX {split} 분할", "",
            "| 부분집합 | CER 감소 (시드 범위) | 악화된 쌍 비율 |", "|---|---|---|",
        ]
        for sname, row in s.items():
            if sname == "per_rule":
                continue
            c, w = row["cer_reduction"], row["worsened_share"]
            if c["mean"] is None:
                continue
            lines.append(f"| {sname} | {c['mean'] * 100:+.2f}%p ({c['min'] * 100:+.2f}~{c['max'] * 100:+.2f}) | "
                         f"{w['mean']:.1%} |")
        lines += ["", "| 규칙 | 분류 | CER 감소 (그 규칙이 포함된 쌍) |", "|---|---|---|"]
        for code, row in s["per_rule"].items():
            if row["mean"] is None:
                continue
            lines.append(f"| `{code}` {KOTOX_RULES[code][0]} | {KOTOX_RULES[code][1]} | {row['mean'] * 100:+.2f}%p |")
        lines.append("")
    lines += [
        "## 해석 제한", "",
        "- KOTOX 행에는 규칙이 2~4개 섞여 있다. 규칙별 수치는 '그 규칙이 포함된 쌍'의 평균이라 다른",
        "  규칙의 영향이 섞인다.",
        "- 길이가 바뀐 쌍은 CER을 편집거리로 잰다. 복원기는 길이를 바꾸지 못하므로 그런 쌍에서는",
        "  구조적으로 일부만 고칠 수 있다.",
        "- KOTOX 문장은 유해 댓글·정상 댓글이다. 인젝션 가드레일 채점과는 무관한 복원 지표만 본다.",
    ]
    return "\n".join(lines) + "\n"


# --- part d: 정답 위치 상한 -----------------------------------------------------

def _oracle_restore(model, obfuscated, cleans) -> list[str]:
    """손상된 자리(원문과 다른 자리)만, 모델의 argmax로 바꾼다. 완벽한 탐지기를 가정한 상한."""
    proba = model._forward(obfuscated)
    out = []
    for r, (obf, clean) in enumerate(zip(obfuscated, cleans)):
        chars = list(obf)
        for c, (a, b) in enumerate(zip(obf, clean)):
            if a == b or c >= model.max_len:
                continue
            slots = list(_input_slots(a))
            for slot in model.unknown_slots:
                slots[slot] = int(proba[slot][r, c].argmax())
            if min(slots) >= 0:
                chars[c] = join(*slots)
        out.append("".join(chars))
    return out


def _site_breakdown(cleans, obfuscated, restored) -> dict:
    damaged = touched = repaired = touched_damaged = harmed = 0
    for clean, obf, rest in zip(cleans, obfuscated, restored):
        for a, b, c in zip(clean, obf, rest):
            is_damaged, is_touched = a != b, b != c
            damaged += is_damaged
            touched += is_touched
            touched_damaged += is_damaged and is_touched
            repaired += is_damaged and c == a
            harmed += (not is_damaged) and is_touched
    return {
        "damaged_sites": damaged,
        "detection_recall": touched_damaged / damaged if damaged else 0.0,
        "detection_precision": touched_damaged / touched if touched else 0.0,
        "correction_accuracy_when_touched": repaired / touched_damaged if touched_damaged else 0.0,
        "repair_rate": repaired / damaged if damaged else 0.0,
        "harmed_sites": harmed,
    }


def part_d(run, args) -> tuple[dict, dict]:
    corpus_path, train_items = _train_items(args)
    run.add_input("corpus", corpus_path)
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(f"seeds/{name}", P.SEED_DIR / name)
    for name in ("benign_dev.jsonl", "benign_test_kotox.jsonl"):
        run.add_input(f"derived/{name}", _derived(name))
    benign = _benign_sets()
    tkey = f"{HEADLINE_TARGET:.3f}"
    metrics: dict = {"models": {}, "evaluations": {}}

    jobs = [("all", None, ALL_TECHNIQUES)] + [(f"single_{t}", t, (t,)) for t in SINGLE_TECHNIQUES]
    for label, single, techniques in jobs:
        eval_sets = _seed_eval_sets(techniques, args)
        for seed in args.seeds:
            key = f"{label}/seed{seed}"
            print(f"[E11d] 모델 {key} ...", flush=True)
            if single is None:
                model, stats = _train_universal(train_items, None, seed, args)
            else:
                pairs = P.make_pairs(train_items, [single], INTENSITIES, gen_seed=args.gen_seed,
                                     include_identity=True)
                model, stats = _fit(single, pairs, seed, label, args)
            metrics["models"][key] = stats
            thresholds = _thresholds(model, benign["dev"])
            benign_rate = M.clean_mutation_rate(
                benign["test_pooled"],
                [r[0] for r in model.restore_many(benign["test_pooled"], thresholds[tkey])],
            )["sentence_rate"]
            per_set = {}
            for name, data in eval_sets.items():
                obf, cleans = data["obfuscated"], data["cleans"]
                actual = [r[0] for r in model.restore_many(obf, thresholds[tkey])]
                free = [r[0] for r in model.restore_many(obf, 0.0)]
                oracle = _oracle_restore(model, obf, cleans)
                per_set[name] = {
                    "at_1pct": {**_cer_rows(cleans, obf, actual, data["clusters"]),
                                **_site_breakdown(cleans, obf, actual)},
                    "no_abstention": {**_cer_rows(cleans, obf, free, data["clusters"]),
                                      **_site_breakdown(cleans, obf, free)},
                    "oracle": {**_cer_rows(cleans, obf, oracle, data["clusters"]),
                               **_site_breakdown(cleans, obf, oracle)},
                }
            metrics["evaluations"][key] = {
                "thresholds": thresholds, "benign_test_pooled_at_1pct": benign_rate, "sets": per_set,
            }
            run.write_json("metrics.json", metrics)

    summary: dict = {}
    for label, _single, techniques in jobs:
        rows = {}
        groups = ({f: FAMILIES[f] for f in FAMILIES} if label == "all" else {techniques[0]: techniques})
        for group, members in groups.items():
            names = [f"{t}@{i}" for t in members for i in INTENSITIES]

            def collect(mode, metric):
                return _mean_range([
                    statistics.fmean(metrics["evaluations"][f"{label}/seed{s}"]["sets"][n][mode][metric]
                                     for n in names)
                    for s in args.seeds])

            actual = collect("at_1pct", "cer_reduction")
            oracle = collect("oracle", "cer_reduction")
            rows[group] = {
                "actual_cer_reduction": actual,
                "oracle_cer_reduction": oracle,
                "actual_detection_recall": collect("at_1pct", "detection_recall"),
                "oracle_correction_accuracy": collect("oracle", "repair_rate"),
                "detection_limited_share": (
                    (oracle["mean"] - actual["mean"]) / oracle["mean"]
                    if oracle["mean"] and oracle["mean"] > 0 else None
                ),
                "oracle_passes": oracle["mean"] is not None and oracle["mean"] >= MIN_CER_REDUCTION,
            }
        summary[label] = rows
    metrics["summary"] = summary
    metrics["status"] = "MEASURED"
    headline = {}
    for label, rows in summary.items():
        for group, row in rows.items():
            headline[f"{label}/{group}/actual"] = row["actual_cer_reduction"]["mean"]
            headline[f"{label}/{group}/oracle"] = row["oracle_cer_reduction"]["mean"]
    return metrics, headline


def report_d(metrics: dict, args) -> str:
    lines = [
        "# E11d — 정답 위치 상한: 완벽한 탐지기를 붙이면 (s02)", "",
        "`oracle`은 원문과 다른 자리만 골라 모델의 예측으로 바꾼다 — 탐지가 완벽하고 정상 자리는",
        "건드리지 않으므로 benign 변형률은 정의상 0이다. `actual`은 benign dev 1% 임계값에서의 실제",
        "성능이다. `탐지 손실 몫` = (oracle − actual) / oracle 은 성능 부족 중 '어디를 고칠지'에서",
        "잃는 몫이다.", "",
        "| 모델 | 기법(군) | actual CER 감소 | oracle CER 감소 | 탐지 손실 몫 | 탐지 재현율(actual) | 정답 위치 교정 정확도 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, rows in metrics["summary"].items():
        for group, row in rows.items():
            a, o = row["actual_cer_reduction"], row["oracle_cer_reduction"]
            share = row["detection_limited_share"]
            lines.append(
                f"| {label} | {group} | {a['mean'] * 100:+.2f}%p | {o['mean'] * 100:+.2f}%p | "
                f"{share:.1%} | {row['actual_detection_recall']['mean']:.1%} | "
                f"{row['oracle_correction_accuracy']['mean']:.1%} |"
                if a["mean"] is not None and o["mean"] is not None and share is not None
                else f"| {label} | {group} | — | — | — | — | — |"
            )
    lines += [
        "", "## 해석 제한", "",
        "- oracle은 상한이다. 실제 탐지기는 정상 자리를 일부 건드리고 손상 자리를 일부 놓친다.",
        "- 교정 정확도는 같은 모델의 argmax를 쓴다. 탐지기를 따로 두면 수정기도 다시 학습할 수 있으므로",
        "  분리 구조의 실제 상한은 이보다 높을 수 있다.",
    ]
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run = R.Run.start(
        run_id=args.run_id,
        experiment=EXPERIMENT,
        config={
            "part": args.part,
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "pair_budget": args.pair_budget,
            "epochs": args.epochs,
            "seeds": list(args.seeds),
            "families": {f: list(ts) for f, ts in FAMILIES.items()},
            "excluded_families": list(args.families),
            "single_techniques": list(SINGLE_TECHNIQUES),
            "intensities": list(INTENSITIES),
            "mutation_targets": list(MUTATION_TARGETS),
            "headline_target": HEADLINE_TARGET,
            "gen_seed": args.gen_seed,
            "device": args.device,
            "architecture": "bilstm(max_len=96, embed=96, hidden=112, layers=2, batch=128)",
            "corpus_license": C.LICENSE_NOTE,
            "kotox_license": "MIT (ssgyejin/KOTOX)",
        },
    )
    part = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[args.part]
    metrics, headline = part(run, args)
    report = {"a": lambda m: report_a(m), "b": lambda m: report_b(m, args),
              "c": lambda m: report_c(m, args), "d": lambda m: report_d(m, args)}[args.part](metrics)
    run.write_text("report.md", report)
    run.finish(metrics, headline={k: v for k, v in headline.items() if v is not None})
    print(f"\n[E11{args.part}] 완료 -> {run.directory}  상태 {metrics.get('status')}")


if __name__ == "__main__":
    main()
