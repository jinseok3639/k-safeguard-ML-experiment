"""P04.11 — 사전학습된 KcELECTRA 판별기가 난독화·남은 오류를 감지하나 (p04.7-discriminator-probe).

ELECTRA 판별기는 사전학습에서 토큰마다 "원래 것 / 바뀐 것"을 맞히도록 배웠다. 학습 없이 그대로 붙여서
(1) 난독화 문장의 망가진 토큰, (2) P04.6 정규화기 복원 결과에 남은 오류 토큰을 짚는지, (3) 학습 데이터에 없던
kotka ㅇ→ㅁ에서도 되는지, (4) 정상 문장에서 얼마나 헛짚는지 본다.

판정(노트에 실행 전 고정): 난독화 토큰 AUC ≥ 0.9 / 남은 오류 토큰 AUC ≥ 0.8 / ㅇ→ㅁ 남은 오류 토큰 AUC ≥ 0.8 /
원문 695 중 1%만 걸리는 문장 임계값에서 남은 오류가 있는 외부 생성기 복원 문장을 50% 이상 잡음.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import evalset as V
from kslab import record as R
from kslab import restore_metrics as M
from kslab.models.pretrained_tagger import KCELECTRA, KCELECTRA_REVISION, PretrainedEditTagger, encoder_path
from restoration.exp.run_p04_3_edit_tagger import EVALSET, MODELS
from restoration.exp.run_p04_6_reeval_metrics import _load, board_of, original

EXPERIMENT = "P04_normalizer"
NORMALIZER = ("p04.3b-kcelectra-ft-s0.pt", "p04.3b-kcelectra-ft", ("kcelectra",))
SQUAREDOG = "kotka/squaredog"
BENIGN_FLAG_BUDGET = 7          # 원문 695 중 1%
BARS = {"obf_token_auc": 0.9, "residual_token_auc": 0.8, "squaredog_residual_auc": 0.8, "residual_sentence_recall": 0.5}


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m{s % 60:02d}s"


class Discriminator:
    """원래 KcELECTRA 판별기. 토큰마다 (글자 구간, 바뀐 토큰일 확률)."""

    def __init__(self, path: str, device: str) -> None:
        import torch
        from transformers import AutoTokenizer, ElectraForPreTraining

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = ElectraForPreTraining.from_pretrained(path).to(device).eval()

    def score(self, texts: list[str], label: str, start: float, batch: int = 64) -> list[list[tuple[int, int, float]]]:
        out = []
        for k in range(0, len(texts), batch):
            enc = self.tokenizer(texts[k:k + batch], return_offsets_mapping=True, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            with self.torch.no_grad():
                logits = self.model(**{n: t.to(self.device) for n, t in enc.items()}).logits
            probs = self.torch.sigmoid(logits.float()).cpu().numpy()
            for row, prob in zip(offsets, probs):
                out.append([(a, b, float(p)) for (a, b), p in zip(row, prob) if b > a])
            done = min(k + batch, len(texts))
            if done == len(texts) or (k // batch) % 50 == 0:
                print(f"  [score {label}] {done}/{len(texts)} ({done / len(texts):.0%}) {_elapsed(start)}", flush=True)
        return out


def token_rows(tokens: list[tuple[int, int, float]], wrong: list[bool]) -> list[tuple[float, bool]]:
    """토큰마다 (점수, 망가졌나). 토큰이 덮는 글자 중 하나라도 어긋났으면 망가진 토큰이다."""
    return [(p, any(wrong[a:b])) for a, b, p in tokens]


def auc(rows: list[tuple[float, bool]]) -> float | None:
    from sklearn.metrics import roc_auc_score

    labels = [w for _, w in rows]
    if all(labels) or not any(labels):
        return None
    return float(roc_auc_score(labels, [p for p, _ in rows]))


def pr_at(rows: list[tuple[float, bool]], threshold: float = 0.5) -> dict:
    hit = sum(1 for p, w in rows if p >= threshold and w)
    flagged = sum(1 for p, _ in rows if p >= threshold)
    wrong = sum(1 for _, w in rows if w)
    return {"precision": hit / flagged if flagged else None, "recall": hit / wrong if wrong else None,
            "tokens": len(rows), "wrong_tokens": wrong}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p04.7-discriminator-probe")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.time()

    file, source_run, path = NORMALIZER
    threshold = original(source_run, path)["threshold"]
    cfg = {"discriminator": KCELECTRA, "discriminator_revision": KCELECTRA_REVISION, "normalizer": file,
           "normalizer_threshold": threshold, "benign_flag_budget": BENIGN_FLAG_BUDGET, "bars": BARS}
    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT, config=cfg)
    run.add_input("evalset_v1/records.jsonl", EVALSET / "records.jsonl")
    records = [r for r in _load(EVALSET / "records.jsonl") if board_of(r) in ("ext", "int")]
    cleans = [s.text for s in V.load_sources()]
    print(f"[1/4] 데이터: 외부 {sum(board_of(r) == 'ext' for r in records)} · 내부 "
          f"{sum(board_of(r) == 'int' for r in records)} · 원문 {len(cleans)} ({_elapsed(start)})", flush=True)

    print(f"[2/4] 정규화기 복원 ({file}, 임계값 {threshold:.3f})", flush=True)
    normalizer = PretrainedEditTagger.load(MODELS / file, device=args.device)
    obfs = [r["obfuscated"] for r in records]
    restored = []
    for k in range(0, len(obfs), 2048):
        chunk = obfs[k:k + 2048]
        restored += [normalizer.decide(t, row, threshold) for t, row in zip(chunk, normalizer.predict(chunk, batch_size=32))]
        print(f"  [restore] {len(restored)}/{len(obfs)} ({len(restored) / len(obfs):.0%}) {_elapsed(start)}", flush=True)
    del normalizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    print(f"[3/4] 판별기 점수 ({KCELECTRA}) {_elapsed(start)}", flush=True)
    disc = Discriminator(encoder_path(), args.device)
    obf_tokens = disc.score(obfs, "난독화", start)
    rest_tokens = disc.score(restored, "복원", start)
    benign_tokens = disc.score(cleans, "원문", start)

    print(f"[4/4] 집계 {_elapsed(start)}", flush=True)
    groups: dict[str, list] = defaultdict(list)       # 이름 → 토큰 행
    sentences: dict[str, list] = defaultdict(list)    # 이름 → (문장 점수, 남은 오류 있나)
    for r, obf, rest, ot, rt in zip(records, obfs, restored, obf_tokens, rest_tokens):
        board = board_of(r)
        o_rows = token_rows(ot, M.wrong_chars(obf, r["clean"]))
        r_wrong = M.wrong_chars(rest, r["clean"])
        r_rows = token_rows(rt, r_wrong)
        names = [board] + ([f"{board}:{r['name'].split(':')[0]}"] if board == "ext" else [])
        for name in names:
            groups[f"obf/{name}"] += o_rows
            groups[f"residual/{name}"] += r_rows
            sentences[f"obf/{name}"].append((max((p for _, _, p in ot), default=0.0), any(M.wrong_chars(obf, r["clean"]))))
            sentences[f"residual/{name}"].append((max((p for _, _, p in rt), default=0.0), any(r_wrong)))
    benign_max = sorted((max((p for _, _, p in t), default=0.0) for t in benign_tokens), reverse=True)
    flag_at = benign_max[BENIGN_FLAG_BUDGET] if len(benign_max) > BENIGN_FLAG_BUDGET else 0.0

    tables = {}
    for name, rows in sorted(groups.items()):
        sent = sentences[name]
        with_error = [s for s, w in sent if w]
        tables[name] = {"token_auc": auc(rows), **pr_at(rows), "sentences": len(sent),
                        "sentences_with_error": len(with_error),
                        "sentence_recall_at_benign_1pct": (sum(s > flag_at for s in with_error) / len(with_error)
                                                           if with_error else None)}
    benign_rows = [(p, False) for t in benign_tokens for _, _, p in t]
    suspicious = sorted(((p, text[a:b]) for text, t in zip(cleans, benign_tokens) for a, b, p in t), reverse=True)[:25]
    benign = {"sentences": len(cleans), "flag_threshold": flag_at,
              "flagged_at_0.5": sum(s >= 0.5 for s in benign_max) / len(benign_max),
              "token_flag_rate_at_0.5": sum(p >= 0.5 for p, _ in benign_rows) / len(benign_rows),
              "top_tokens": [[round(p, 3), w] for p, w in suspicious]}

    sq = f"residual/ext:{SQUAREDOG}"
    checks = {
        "obf_token_auc": tables["obf/ext"]["token_auc"],
        "residual_token_auc": tables["residual/ext"]["token_auc"],
        "squaredog_residual_auc": tables.get(sq, {}).get("token_auc"),
        "residual_sentence_recall": tables["residual/ext"]["sentence_recall_at_benign_1pct"],
    }
    passed = {k: v is not None and v >= BARS[k] for k, v in checks.items()}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if any(f.name == "Malgun Gothic" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name, title in zip(axes, ("obf/ext", "residual/ext", sq),
                               ("외부 생성기 난독화 문장", "외부 생성기 복원 결과", "kotka ㅇ→ㅁ 복원 결과")):
        rows = groups.get(name, [])
        bins = np.linspace(0, 1, 41)
        ax.hist([p for p, w in rows if not w], bins=bins, alpha=0.6, density=True, label="맞는 토큰")
        ax.hist([p for p, w in rows if w], bins=bins, alpha=0.6, density=True, label="어긋난 토큰")
        ax.set_title(f"{title} (AUC {tables.get(name, {}).get('token_auc') or float('nan'):.3f})")
        ax.set_xlabel("판별기 점수(바뀐 토큰일 확률)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(run.directory / "scores.png", dpi=120)
    plt.close(fig)

    def f3(x):
        return "-" if x is None else f"{x:.3f}"

    lines = [f"# {args.run_id} — 원래 KcELECTRA 판별기의 감지력", "",
             "| 기준 | 목표 | 결과 | 판정 |", "|---|---|---|---|"]
    labels = {"obf_token_auc": "난독화 토큰 AUC(외부)", "residual_token_auc": "남은 오류 토큰 AUC(외부 복원)",
              "squaredog_residual_auc": "ㅇ→ㅁ 남은 오류 토큰 AUC",
              "residual_sentence_recall": "원문 1%만 걸리는 임계값에서 남은 오류 문장 감지율"}
    for key, value in checks.items():
        lines.append(f"| {labels[key]} | ≥ {BARS[key]} | {f3(value)} | {'통과' if passed[key] else '미달'} |")
    lines += ["", f"문장 임계값(원문 695 중 {BENIGN_FLAG_BUDGET}문장만 넘음): {flag_at:.3f}. "
                  f"점수 0.5 이상 토큰이 있는 원문: {benign['flagged_at_0.5']:.1%}", "",
              "| 묶음 | 토큰 AUC | 정밀도@0.5 | 재현율@0.5 | 어긋난 토큰 / 전체 | 오류 문장 감지율(원문 1%) |",
              "|---|---|---|---|---|---|"]
    for name, t in tables.items():
        lines.append(f"| {name} | {f3(t['token_auc'])} | {f3(t['precision'])} | {f3(t['recall'])} | "
                     f"{t['wrong_tokens']} / {t['tokens']} | {f3(t['sentence_recall_at_benign_1pct'])} |")
    lines += ["", "## 원문 695에서 가장 의심받은 토큰", "",
              ", ".join(f"`{w}` {p}" for p, w in benign["top_tokens"]), "", "그림: scores.png"]
    run.write_text("report.md", "\n".join(lines) + "\n")
    metrics = {"checks": checks, "passed": passed, "groups": tables, "benign": benign,
               "status": "pass" if all(passed.values()) else "fail"}
    run.finish(metrics, headline={k: v for k, v in checks.items() if v is not None})
    print("\n".join(lines))
    print(f"\n[{args.run_id}] {metrics['status']} -> {run.directory} ({_elapsed(start)})")


if __name__ == "__main__":
    main()
