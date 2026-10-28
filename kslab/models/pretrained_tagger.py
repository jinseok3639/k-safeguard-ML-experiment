"""사전학습 인코더를 얹은 글자 편집 태거 — P04.3b(전체 미세조정)·P04.3c(LoRA).

`EditTagger`와 출력(글자별 유지·삭제·교체 + 앞 공백)과 기권 방식이 같다. 다른 점은 입력이다.

    난독화 문장 ─▶ 사전학습 인코더(KcELECTRA 등) ─▶ 토큰 벡터를 그 토큰에 속한 글자들에 나눠 줌 ─┐
               └─▶ 글자 + 자모 임베딩 ─────────────────────────────────────────────┴─▶ BiLSTM 1층 ─▶ 출력

인코더는 단어 조각(토큰) 단위로 읽어서 한 토큰 안의 글자들이 같은 벡터를 받는다. 글자·자모 임베딩이 그 글자들을
서로 구별한다. 토큰에 속하지 않는 글자(공백, 길이 상한 밖)는 토큰 벡터 자리를 0으로 둔다.

`lora=True`면 인코더 원래 가중치는 얼리고 어텐션의 query·value에 LoRA 보정만 학습한다(`peft`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..jamo import N_CHO, N_JONG, N_JUNG
from .edit_tagger import OPS, PAD, EditTagger

KCELECTRA = "beomi/KcELECTRA-base-v2022"
KCELECTRA_REVISION = "87285578b4686b6d190abd40e947a2a416add6de"
LORA_TARGETS = ("query", "value")


def encoder_path(repo: str = KCELECTRA, revision: str = KCELECTRA_REVISION) -> str:
    """받아 둔 스냅샷 경로(`HF_HUB_CACHE` 아래). 없으면 받지 않고 멈춘다(다운로드는 따로 승인받아 한다)."""
    import os
    from pathlib import Path

    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "")) / "hub")
    path = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"{repo}@{revision[:7]} 스냅샷이 없습니다: {path}")
    return str(path)


def _pretrained_net(vocab_size: int, cfg: "PretrainedEditTagger"):
    import torch
    from transformers import AutoModel

    nn = torch.nn
    encoder = AutoModel.from_pretrained(cfg.encoder)
    if cfg.lora:
        from peft import LoraConfig, get_peft_model

        encoder = get_peft_model(encoder, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                                     lora_dropout=cfg.dropout, target_modules=list(LORA_TARGETS)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.proj = nn.Linear(encoder.config.hidden_size, cfg.proj_dim)
            self.char = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD)
            self.slot = nn.ModuleList(nn.Embedding(n + 1, cfg.jamo_dim, padding_idx=0) for n in (N_CHO, N_JUNG, N_JONG))
            width = cfg.proj_dim + cfg.embed_dim + 3 * cfg.jamo_dim
            self.norm_in = nn.LayerNorm(width)
            self.lstm = nn.LSTM(width, cfg.hidden, num_layers=cfg.layers, batch_first=True, bidirectional=True)
            self.norm_out = nn.LayerNorm(cfg.hidden * 2)
            self.drop = nn.Dropout(cfg.dropout)
            h = cfg.hidden * 2
            self.op = nn.Linear(h, len(OPS))
            self.space = nn.Linear(h, 2)
            self.slots = nn.ModuleList(nn.Linear(h, n) for n in (N_CHO, N_JUNG, N_JONG))

        def forward(self, chars, slots, keys, input_ids, attention, char2tok, covered):
            tokens = self.encoder(input_ids=input_ids, attention_mask=attention).last_hidden_state
            index = char2tok.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
            per_char = tokens.gather(1, index) * covered.unsqueeze(-1).to(tokens.dtype)
            parts = [self.proj(per_char), self.char(chars)] + [emb(slots[..., k]) for k, emb in enumerate(self.slot)]
            x = self.norm_in(torch.cat(parts, dim=-1))
            out, _ = self.lstm(x)
            out = self.drop(self.norm_out(out))
            return self.op(out), self.space(out), [head(out) for head in self.slots]

    return Net()


@dataclass
class PretrainedEditTagger(EditTagger):
    encoder: str = ""
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    encoder_lr: float = 3e-5
    proj_dim: int = 256
    hidden: int = 128
    layers: int = 1
    epochs: int = 3
    batch_size: int = 32
    max_tokens: int = 512

    CONFIG_KEYS = EditTagger.CONFIG_KEYS + ("encoder", "lora", "lora_r", "lora_alpha", "encoder_lr", "proj_dim")

    def __post_init__(self):
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.encoder)
        return self._tokenizer

    def _build_net(self):
        return _pretrained_net(len(self.vocab) + 2, self)

    def _param_groups(self) -> list[dict]:
        enc = [p for n, p in self.net.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        rest = [p for n, p in self.net.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        return [{"params": enc, "lr": self.encoder_lr}, {"params": rest, "lr": self.lr}]

    def _features(self, texts):
        chars, slots, keys = super()._features(texts)
        width = chars.shape[1]
        enc = self.tokenizer(list(texts), return_offsets_mapping=True, padding=True, truncation=True,
                             max_length=self.max_tokens, return_tensors="np")
        char2tok = np.zeros((len(texts), width), dtype=np.int64)
        covered = np.zeros((len(texts), width), dtype=np.int64)
        for r, offsets in enumerate(enc["offset_mapping"]):
            for k, (a, b) in enumerate(offsets):
                if b > a:                      # 특수 토큰·패딩은 (0, 0)
                    char2tok[r, a:min(b, width)] = k
                    covered[r, a:min(b, width)] = 1
        return (chars, slots, keys, enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64),
                char2tok, covered)

    def trainable_parameters(self) -> tuple[int, int]:
        params = list(self.net.parameters())
        return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)
