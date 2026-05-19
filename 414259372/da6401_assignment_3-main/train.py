"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            fill = self.smoothing / max(self.vocab_size - 2, 1)
            true_dist.fill_(fill)
            true_dist[:, self.pad_idx] = 0.0
            target_mask = target != self.pad_idx
            if target_mask.any():
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
                true_dist[target == self.pad_idx] = 0.0
        loss = self.criterion(log_probs, true_dist)
        denom = (target != self.pad_idx).sum().clamp_min(1)
        return loss / denom


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    for batch in data_iter:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            src, tgt = batch
        else:
            raise ValueError("Expected each batch to be (src, tgt)")

        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        vocab_size = logits.size(-1)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            non_pad = (tgt_out != getattr(model, "pad_idx", 1)).sum().item()
            total_tokens += max(non_pad, 1)
            total_loss += loss.item() * max(non_pad, 1)

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=getattr(model, "pad_idx", 1))
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
    return ys


def _tensor_to_tokens(seq, vocab) -> list[str]:
    tokens = []
    for idx in seq:
        if idx in {0, 1, 2, 3}:
            continue
        if hasattr(vocab, "lookup_token"):
            tok = vocab.lookup_token(int(idx))
        elif hasattr(vocab, "itos"):
            tok = vocab.itos[int(idx)]
        else:
            tok = str(idx)
        if tok not in {"<pad>", "<sos>", "<eos>"}:
            tokens.append(tok)
    return tokens


def _bleu_score(references: list[list[str]], hypotheses: list[list[str]], max_n: int = 4) -> float:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    precisions = []
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = ngrams(ref, n)
            hyp_counts = ngrams(hyp, n)
            total += max(len(hyp) - n + 1, 0)
            for ng, count in hyp_counts.items():
                matches += min(count, ref_counts.get(ng, 0))
        precisions.append((matches + 1e-9) / (total + 1e-9))
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * bp * geo_mean


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()
    hypotheses = []
    references = []
    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = make_src_mask(src)
        for i in range(src.size(0)):
            pred = greedy_decode(
                model,
                src[i:i+1],
                src_mask[i:i+1],
                max_len=max_len,
                start_symbol=getattr(model, "sos_idx", 2),
                end_symbol=getattr(model, "eos_idx", 3),
                device=device,
            )[0].tolist()
            hyp_tokens = _tensor_to_tokens(pred, tgt_vocab)
            ref_tokens = _tensor_to_tokens(tgt[i].tolist(), tgt_vocab)
            hypotheses.append(hyp_tokens)
            references.append(ref_tokens)
    return _bleu_score(references, hypotheses)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": {
                "src_vocab_size": model.src_vocab_size,
                "tgt_vocab_size": model.tgt_vocab_size,
                "d_model": model.d_model,
                "N": model.N,
                "num_heads": model.num_heads,
                "d_ff": model.d_ff,
                "dropout": model.dropout,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0))


def run_training_experiment() -> None:
    """Minimal end-to-end training scaffold."""
    try:
        from dataset import Multi30kDataset
        from lr_scheduler import NoamScheduler
    except Exception as exc:
        raise RuntimeError("Required modules are unavailable") from exc

    train_ds = Multi30kDataset(split="train")
    src_vocab, tgt_vocab = train_ds.build_vocab()
    data = train_ds.process_data()

    def collate(batch):
        src, tgt = zip(*batch)
        max_src = max(len(x) for x in src)
        max_tgt = max(len(x) for x in tgt)
        pad = 1
        src_t = torch.full((len(batch), max_src), pad, dtype=torch.long)
        tgt_t = torch.full((len(batch), max_tgt), pad, dtype=torch.long)
        for i, (s, t) in enumerate(zip(src, tgt)):
            src_t[i, :len(s)] = torch.tensor(s)
            tgt_t[i, :len(t)] = torch.tensor(t)
        return src_t, tgt_t

    loader = DataLoader(list(zip([x for x, _ in data], [y for _, y in data])), batch_size=2, collate_fn=collate)
    model = Transformer(len(src_vocab), len(tgt_vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=model.d_model, warmup_steps=4000)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1, smoothing=0.1)
    run_epoch(loader, model, loss_fn, optimizer, scheduler, is_train=True)


if __name__ == "__main__":
    run_training_experiment()
