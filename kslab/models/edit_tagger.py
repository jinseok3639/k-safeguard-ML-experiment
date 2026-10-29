"""글자 편집 태거 (torch) — P04 정규화기 본선(후보 A).

입력 글자마다 `kslab.edit_tags`의 태그를 예측한다: 유지·삭제·음절 교체(초·중·종 슬롯) + 앞 공백 넣기. 1기 슬롯 복원기와
달리 후보 위치 규칙 없이 모든 글자를 보고, 길이가 바뀌는 난독화(띄어쓰기·기호·자모 분해)도 되돌린다.

입력 특징은 세 가지를 더한다.
- 글자 임베딩(학습 데이터에 두 번 이상 나온 글자, 나머지는 UNK)
- 자모 임베딩: 음절은 (초, 중, 종), 낱자 자모는 해당 슬롯 하나. 처음 보는 음절도 자모로 읽게 한다.
- (선택) 음운 대표형 임베딩: 된소리·거센소리 → 예사소리, 혼동 모음·반모음 → 대표 모음, 받침 → 대표 받침. 소리가 비슷한
  글자를 같은 키로 묶는다(윤태진 2010, RobEn, ANTHRO). `phonetic=True`일 때만 켠다.

기권: 편집마다 확신도를 내고(삭제는 P(삭제), 교체는 P(교체)×슬롯 확률 곱, 공백은 P(공백)) `threshold` 아래면 손대지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .. import edit_tags as E
from .. import phonology as PH
from ..jamo import CHO_INDEX, JONG_INDEX, JUNG_INDEX, N_CHO, N_JONG, N_JUNG, is_syllable, join, split
from . import device as _device

OPS = (E.KEEP, E.DELETE, E.SUB)
OP_INDEX = {op: i for i, op in enumerate(OPS)}
IGNORE = -100
PAD, UNK = 0, 1

_TENSE_ASPIRATE = {"ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄸ": "ㄷ", "ㅌ": "ㄷ", "ㅃ": "ㅂ", "ㅍ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ", "ㅊ": "ㅈ"}
_VOWEL_REP = {"ㅑ": "ㅏ", "ㅘ": "ㅏ", "ㅕ": "ㅓ", "ㅝ": "ㅓ", "ㅛ": "ㅗ", "ㅠ": "ㅜ", "ㅟ": "ㅣ", "ㅢ": "ㅣ",
              "ㅐ": "ㅔ", "ㅒ": "ㅔ", "ㅖ": "ㅔ", "ㅚ": "ㅞ", "ㅙ": "ㅞ"}
_CODA_REP = {**PH.CLUSTER_REDUCE, **PH.CODA_NEUTRALIZE}


def _rep_table(table: dict, index: dict, size: int) -> list[int]:
    """슬롯 인덱스 -> 대표 자모의 인덱스. 표에 없으면 자기 자신."""
    char_of = {i: ch for ch, i in index.items()}
    return [index[table[char_of[i]]] if char_of.get(i) in table else i for i in range(size)]


_CHO_REP = _rep_table(_TENSE_ASPIRATE, CHO_INDEX, N_CHO)
_JUNG_REP = _rep_table(_VOWEL_REP, JUNG_INDEX, N_JUNG)
_JONG_REP = _rep_table(_CODA_REP, JONG_INDEX, N_JONG)


def jamo_slots(char: str) -> tuple[int, int, int]:
    """(초, 중, 종) 인덱스 + 1. 0은 '없음'. 낱자 자모는 해당 슬롯 하나만 채운다."""
    if is_syllable(char):
        c, j, g = split(char)
        return c + 1, j + 1, g + 1
    if char in CHO_INDEX:
        return CHO_INDEX[char] + 1, 0, 0
    if char in JUNG_INDEX:
        return 0, JUNG_INDEX[char] + 1, 0
    if char in JONG_INDEX:
        return 0, 0, JONG_INDEX[char] + 1
    return 0, 0, 0


def phonetic_key(char: str) -> tuple[int, int, int]:
    """음운 대표형 (초, 중, 종) + 1. 음절이 아니면 자모 슬롯과 같은 규칙으로 하나만 채운다."""
    c, j, g = jamo_slots(char)
    return (_CHO_REP[c - 1] + 1 if c else 0, _JUNG_REP[j - 1] + 1 if j else 0, _JONG_REP[g - 1] + 1 if g else 0)


def _net(vocab_size: int, cfg: "EditTagger"):
    import torch
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.char = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD)
            self.slot = nn.ModuleList(nn.Embedding(n + 1, cfg.jamo_dim, padding_idx=0) for n in (N_CHO, N_JUNG, N_JONG))
            self.key = nn.ModuleList(nn.Embedding(n + 1, cfg.jamo_dim, padding_idx=0) for n in (N_CHO, N_JUNG, N_JONG)) \
                if cfg.phonetic else None
            width = cfg.embed_dim + 3 * cfg.jamo_dim * (2 if cfg.phonetic else 1)
            self.norm_in = nn.LayerNorm(width)
            self.lstm = nn.LSTM(width, cfg.hidden, num_layers=cfg.layers, batch_first=True, bidirectional=True,
                                dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.norm_out = nn.LayerNorm(cfg.hidden * 2)
            self.drop = nn.Dropout(cfg.dropout)
            h = cfg.hidden * 2
            self.op = nn.Linear(h, len(OPS))
            self.space = nn.Linear(h, 2)
            self.slots = nn.ModuleList(nn.Linear(h, n) for n in (N_CHO, N_JUNG, N_JONG))

        def forward(self, chars, slots, keys):
            parts = [self.char(chars)] + [emb(slots[..., k]) for k, emb in enumerate(self.slot)]
            if self.key is not None:
                parts += [emb(keys[..., k]) for k, emb in enumerate(self.key)]
            x = self.norm_in(torch.cat(parts, dim=-1))
            out, _ = self.lstm(x)
            out = self.drop(self.norm_out(out))
            return self.op(out), self.space(out), [head(out) for head in self.slots]

    return Net()


@dataclass
class EditTagger:
    phonetic: bool = False
    embed_dim: int = 128
    jamo_dim: int = 32
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    max_len: int = 384
    random_state: int = 2026
    device: str = "cpu"
    amp: bool = False                 # CUDA에서 bf16 자동 혼합 정밀도
    batch_chars: int | None = None    # 배치당 글자 수 상한(긴 문장 배치를 작게). None이면 batch_size만 본다
    vocab: dict = field(default_factory=dict)
    net: object | None = None
    history: list = field(default_factory=list)

    # --- 인코딩 -------------------------------------------------------------
    def _fit_vocab(self, texts: Sequence[str], min_count: int = 2) -> None:
        counts: dict[str, int] = {}
        for t in texts:
            for ch in t:
                counts[ch] = counts.get(ch, 0) + 1
        chars = sorted(ch for ch, n in counts.items() if n >= min_count)
        self.vocab = {ch: i + 2 for i, ch in enumerate(chars)}

    def _features(self, texts: Sequence[str]):
        width = min(max((len(t) for t in texts), default=1), self.max_len) or 1
        chars = np.zeros((len(texts), width), dtype=np.int64)
        slots = np.zeros((len(texts), width, 3), dtype=np.int64)
        keys = np.zeros((len(texts), width, 3), dtype=np.int64)
        for r, t in enumerate(texts):
            for c, ch in enumerate(t[:width]):
                chars[r, c] = self.vocab.get(ch, UNK)
                slots[r, c] = jamo_slots(ch)
                if self.phonetic:
                    keys[r, c] = phonetic_key(ch)
        return chars, slots, keys

    @staticmethod
    def _labels(tag_rows: Sequence[list], width: int):
        op = np.full((len(tag_rows), width), IGNORE, dtype=np.int64)
        space = np.full((len(tag_rows), width), IGNORE, dtype=np.int64)
        slot = np.full((len(tag_rows), width, 3), IGNORE, dtype=np.int64)
        for r, tags in enumerate(tag_rows):
            for i, t in enumerate(tags[:width]):
                op[r, i] = OP_INDEX[t.op]
                space[r, i] = int(t.space_before)
                if t.op == E.SUB:
                    slot[r, i] = split(t.char)
        return op, space, slot

    # --- 하위 클래스가 바꾸는 자리 ------------------------------------------
    def _build_net(self):
        return _net(len(self.vocab) + 2, self)

    def _param_groups(self) -> list[dict]:
        return [{"params": [p for p in self.net.parameters() if p.requires_grad], "lr": self.lr}]

    def _autocast(self, dev):
        import contextlib

        import torch

        if self.amp and dev.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _batches(self, lengths: Sequence[int]) -> list[np.ndarray]:
        """길이가 비슷한 문장끼리 묶어 패딩을 줄인다. `batch_chars`가 있으면 긴 문장 배치는 문장 수를 줄인다."""
        order = np.argsort(lengths, kind="stable")
        if self.batch_chars is None:
            return [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        batches, cur = [], []
        for i in order:
            width = min(lengths[i], self.max_len)
            if cur and (len(cur) >= self.batch_size or width * (len(cur) + 1) > self.batch_chars):
                batches.append(np.array(cur))
                cur = []
            cur.append(i)
        if cur:
            batches.append(np.array(cur))
        return batches

    # --- 학습 ---------------------------------------------------------------
    def fit(self, obfs: Sequence[str], cleans: Sequence[str], log=print) -> "EditTagger":
        import torch

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        dev = _device.resolve(self.device)
        self._fit_vocab(list(obfs) + list(cleans))
        gold = [E.oracle_tags(o, c)[0] for o, c in zip(obfs, cleans)]   # 정답 태그는 한 번만 정렬한다
        self.net = self._build_net().to(dev)
        batches = self._batches([len(o) for o in obfs])
        groups = self._param_groups()
        opt = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                    total_steps=self.epochs * len(batches))
        ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
        # 배치 구성은 에폭마다 같고 순서만 섞으므로 특징·레이블을 한 번만 만든다(int32로 메모리 절약)
        cache = []
        for idx in batches:
            feats = self._features([obfs[i] for i in idx])
            labels = self._labels([gold[i] for i in idx], feats[0].shape[1])
            cache.append((tuple(a.astype(np.int32) for a in feats), tuple(a.astype(np.int32) for a in labels)))
        t = lambda a: torch.from_numpy(a.astype(np.int64)).to(dev)
        for epoch in range(self.epochs):
            self.net.train()
            total, n = 0.0, 0
            for b in rng.permutation(len(batches)):
                idx = batches[b]
                feats, (op, space, slot) = cache[b]
                with self._autocast(dev):
                    op_logit, sp_logit, slot_logits = self.net(*map(t, feats))
                op_logit, sp_logit = op_logit.float(), sp_logit.float()
                loss = ce(op_logit.flatten(0, 1), t(op).flatten()) + ce(sp_logit.flatten(0, 1), t(space).flatten())
                for k, logit in enumerate(slot_logits):
                    loss = loss + ce(logit.float().flatten(0, 1), t(slot[..., k]).flatten())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(idx)
                n += len(idx)
            self.history.append({"epoch": epoch + 1, "loss": total / n})
            log(f"  epoch {epoch + 1}/{self.epochs} loss {total / n:.4f}")
        return self

    # --- 추론 ---------------------------------------------------------------
    def predict(self, texts: Sequence[str], batch_size: int = 128) -> list[list[tuple]]:
        """문장마다 글자별 (태그, 확신도, 공백 확신도) — 태그는 최고 확률 태그, 기권 전."""
        import torch

        dev = next(self.net.parameters()).device
        self.net.eval()
        out: list[list[tuple]] = [None] * len(texts)
        order = np.argsort([len(t) for t in texts], kind="stable")
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                batch = [texts[i] for i in idx]
                feats = self._features(batch)
                width = feats[0].shape[1]
                with self._autocast(dev):
                    op_logit, sp_logit, slot_logits = self.net(*(torch.from_numpy(a).to(dev) for a in feats))
                op_p = op_logit.float().softmax(-1).cpu().numpy()
                sp_p = sp_logit.float().softmax(-1)[..., 1].cpu().numpy()
                sl = [l.float().softmax(-1).cpu().numpy() for l in slot_logits]
                for r, i in enumerate(idx):
                    row = []
                    for c, ch in enumerate(texts[i][:width]):
                        k = int(op_p[r, c].argmax())
                        if OPS[k] == E.SUB:
                            cj = [int(sl[q][r, c].argmax()) for q in range(3)]
                            conf = float(op_p[r, c, k] * np.prod([sl[q][r, c, cj[q]] for q in range(3)]))
                            sub = join(*cj)
                            tag = E.Tag(E.KEEP) if sub == ch else E.Tag(E.SUB, sub)
                        else:
                            tag, conf = E.Tag(OPS[k]), float(op_p[r, c, k])
                        row.append((tag, conf, float(sp_p[r, c])))
                    row += [(E.Tag(E.KEEP), 1.0, 0.0)] * (len(texts[i]) - len(row))   # max_len 밖은 그대로
                    out[i] = row
        return out

    @staticmethod
    def decide(text: str, row: list[tuple], threshold: float) -> str:
        """확신도가 `threshold` 이상인 편집만 적용한다."""
        tags = []
        for tag, conf, sp in row:
            op = tag if (tag.op != E.KEEP and conf >= threshold) else E.Tag(E.KEEP)
            tags.append(E.Tag(op.op, op.char, space_before=sp >= max(threshold, 0.5)))
        return E.apply_tags(text, tags)

    @staticmethod
    def edit_confidence(row: list[tuple]) -> float:
        """문장에서 가장 확신하는 편집의 확신도. 0이면 아무것도 안 바꾼다(정상 문장 오변경 판단용)."""
        best = 0.0
        for tag, conf, sp in row:
            if tag.op != E.KEEP:
                best = max(best, conf)
            if sp >= 0.5:
                best = max(best, sp)
        return best

    CONFIG_KEYS = ("phonetic", "embed_dim", "jamo_dim", "hidden", "layers", "dropout", "max_len", "random_state")

    def save(self, path) -> None:
        import torch

        cfg = {k: getattr(self, k) for k in self.CONFIG_KEYS}
        torch.save({"config": cfg, "vocab": self.vocab, "state": self.net.state_dict(), "history": self.history}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "EditTagger":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**ck["config"], device=device)
        model.vocab, model.history = ck["vocab"], ck["history"]
        model.net = model._build_net()
        model.net.load_state_dict(ck["state"])
        model.net.to(_device.resolve(device))
        return model
