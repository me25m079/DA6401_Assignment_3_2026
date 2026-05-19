
import os, torch, gdown
BEST_PTH_GDRIVE_ID="1kZHbQhUkjv47j2GgDuDUGnqdZLqrB8CA"

def _load_best_weights(model):
    if BEST_PTH_GDRIVE_ID!="1kZHbQhUkjv47j2GgDuDUGnqdZLqrB8CA" and not os.path.exists("best.pth"):
        gdown.download(
            f"https://drive.google.com/uc?id={BEST_PTH_GDRIVE_ID}",
            "best.pth",
            quiet=False
        )
    if os.path.exists("best.pth"):
        ckpt=torch.load("best.pth", map_location="cpu")
        state=ckpt["model_state_dict"] if isinstance(ckpt,dict) and "model_state_dict" in ckpt else ckpt
        if any(k.startswith("module.") for k in state.keys()):
            state={k.replace("module.","",1):v for k,v in state.items()}
        model.load_state_dict(state, strict=False)

"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from pathlib import Path
from typing import Optional, Tuple, Any

try:
    import gdown
except Exception:
    gdown = None
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.to(torch.bool), torch.finfo(scores.dtype).min)
    attn_w = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_w, V)
    return out, attn_w


# ══════════════════════════════════════════════════════════════════════
#   MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Padding mask for source tokens: [B, 1, 1, S]."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Padding + causal look-ahead mask for decoder: [B, 1, T, T]."""
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    subsequent_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool), diagonal=1
    ).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    return pad_mask | subsequent_mask


# ══════════════════════════════════════════════════════════════════════
#   MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [B, H, T, d_k]

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if mask is not None:
            if mask.dim() == 4:
                mask = mask.expand(q.size(0), self.num_heads, mask.size(-2), mask.size(-1))
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_out = self._combine_heads(attn_out)
        return self.w_o(self.dropout(attn_out))


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#   FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#   ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#   ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   SIMPLE VOCAB + TOKENIZER UTILITIES
# ══════════════════════════════════════════════════════════════════════

class _SimpleVocab:
    def __init__(self, tokens: Optional[list[str]] = None) -> None:
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}
        for tok in ["<unk>", "<pad>", "<sos>", "<eos>"] + (tokens or []):
            self.add_token(tok)

    def add_token(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, index: int) -> str:
        return self.itos[index] if 0 <= index < len(self.itos) else "<unk>"

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def __contains__(self, token: str) -> bool:
        return token in self.stoi


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int | None = None,
        tgt_vocab_size: int | None = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        self.root_dir = Path(__file__).resolve().parent
        self.checkpoint_path = checkpoint_path
        self._weights_loaded = False

        inferred_cfg = self._infer_checkpoint_config(checkpoint_path)
        if inferred_cfg is not None:
            src_vocab_size = src_vocab_size or inferred_cfg.get("src_vocab_size")
            tgt_vocab_size = tgt_vocab_size or inferred_cfg.get("tgt_vocab_size")
            d_model = inferred_cfg.get("d_model", d_model)
            N = inferred_cfg.get("N", N)
            num_heads = inferred_cfg.get("num_heads", num_heads)
            d_ff = inferred_cfg.get("d_ff", d_ff)
            dropout = inferred_cfg.get("dropout", dropout)
            self.d_model = d_model
            self.N = N
            self.num_heads = num_heads
            self.d_ff = d_ff
            self.dropout = dropout

        self.src_vocab_size = src_vocab_size or 37000
        self.tgt_vocab_size = tgt_vocab_size or 37000

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, self.tgt_vocab_size)

        self.src_tokenizer, self.tgt_tokenizer = self._load_tokenizers()
        self.src_vocab, self.tgt_vocab = self._load_vocabularies(src_vocab_size=self.src_vocab_size, tgt_vocab_size=self.tgt_vocab_size)

        self._maybe_load_weights(checkpoint_path)

    # ---- internal helpers -------------------------------------------------

    def _resolve_file(self, candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            path = Path(candidate)
            if path.is_file():
                return str(path)
            path = self.root_dir / candidate
            if path.is_file():
                return str(path)
        return None

    def _infer_checkpoint_config(self, checkpoint_path: str | None) -> Optional[dict[str, Any]]:
        candidates = []
        if checkpoint_path:
            candidates.append(checkpoint_path)
        env_path = os.environ.get("TRANSFORMER_CHECKPOINT_PATH")
        if env_path:
            candidates.append(env_path)
        candidates += [
            str(self.root_dir / name)
            for name in ["checkpoint.pt", "checkpoint.pth", "best_model.pt", "best_model.pth", "transformer.pt", "transformer.pth"]
        ]
        for path in candidates:
            if not path or not os.path.isfile(path):
                continue
            try:
                obj = torch.load(path, map_location="cpu")
            except Exception:
                continue
            if isinstance(obj, dict):
                cfg = obj.get("model_config")
                if isinstance(cfg, dict):
                    return cfg
                state_dict = obj.get("model_state_dict", obj if all(hasattr(v, "shape") for v in obj.values()) else None)
                if isinstance(state_dict, dict):
                    inferred = {}
                    for k, v in state_dict.items():
                        if not torch.is_tensor(v):
                            continue
                        if k.endswith("src_embedding.weight"):
                            inferred["src_vocab_size"] = v.shape[0]
                            inferred["d_model"] = v.shape[1]
                        elif k.endswith("tgt_embedding.weight"):
                            inferred["tgt_vocab_size"] = v.shape[0]
                            inferred["d_model"] = v.shape[1]
                        elif k.endswith("generator.weight"):
                            inferred.setdefault("tgt_vocab_size", v.shape[0])
                            inferred.setdefault("d_model", v.shape[1])
                    if inferred:
                        return inferred
        return None

    def _load_tokenizers(self):
        try:
            import spacy
            src_tok = spacy.blank("de")
            tgt_tok = spacy.blank("en")
            return src_tok, tgt_tok
        except Exception:
            return None, None

    def _load_vocabularies(self, src_vocab_size: int, tgt_vocab_size: int):
        src_vocab_path = self._resolve_file(["src_vocab.json", "de_vocab.json", "vocab_de.json", "src_vocab.txt"])
        tgt_vocab_path = self._resolve_file(["tgt_vocab.json", "en_vocab.json", "vocab_en.json", "tgt_vocab.txt"])

        def load_simple_vocab(path: Optional[str], target_size: int) -> _SimpleVocab:
            vocab = _SimpleVocab()
            if path is None:
                return vocab
            try:
                import json
                if path.endswith(".json"):
                    data = json.load(open(path, "r", encoding="utf-8"))
                    tokens = data if isinstance(data, list) else data.get("itos", [])
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        tokens = [line.strip() for line in f if line.strip()]
                for tok in tokens:
                    vocab.add_token(tok)
                return vocab
            except Exception:
                return vocab

        src_vocab = load_simple_vocab(src_vocab_path, src_vocab_size)
        tgt_vocab = load_simple_vocab(tgt_vocab_path, tgt_vocab_size)
        return src_vocab, tgt_vocab

    def _maybe_load_weights(self, checkpoint_path: str | None) -> None:
        candidates = []
        if checkpoint_path:
            candidates.append(checkpoint_path)
        env_path = os.environ.get("TRANSFORMER_CHECKPOINT_PATH")
        if env_path:
            candidates.append(env_path)
        candidates += [
            str(self.root_dir / name)
            for name in ["checkpoint.pt", "checkpoint.pth", "best_model.pt", "best_model.pth", "transformer.pt", "transformer.pth"]
        ]
        loaded_path = None
        for path in candidates:
            if path and os.path.isfile(path):
                loaded_path = path
                break
        if loaded_path is None and checkpoint_path is not None:
            # Download only if an explicit Google Drive id/URL is provided via env.
            if gdown is not None:
                drive_id = os.environ.get("TRANSFORMER_GDRIVE_ID")
                drive_url = os.environ.get("TRANSFORMER_GDRIVE_URL")
                if drive_url:
                    gdown.download(url=drive_url, output=checkpoint_path, quiet=False, fuzzy=True)
                    loaded_path = checkpoint_path
                elif drive_id:
                    gdown.download(id=drive_id, output=checkpoint_path, quiet=False)
                    loaded_path = checkpoint_path

        if loaded_path is None:
            return

        try:
            obj = torch.load(loaded_path, map_location="cpu")
            if isinstance(obj, dict) and "model_state_dict" in obj:
                state_dict = obj["model_state_dict"]
            else:
                state_dict = obj
            self.load_state_dict(state_dict, strict=False)
            self._weights_loaded = True
        except Exception:
            self._weights_loaded = False

    def _tokenize(self, sentence: str, lang: str = "de") -> list[str]:
        tokenizer = self.src_tokenizer if lang == "de" else self.tgt_tokenizer
        if tokenizer is None:
            return sentence.strip().split()
        try:
            return [tok.text for tok in tokenizer.tokenizer(sentence.strip())]
        except Exception:
            return sentence.strip().split()

    def _numericalize(self, tokens: list[str], vocab: _SimpleVocab) -> list[int]:
        return [self.sos_idx] + [vocab.lookup_index(tok) for tok in tokens] + [self.eos_idx]

    def _detokenize(self, tokens: list[str]) -> str:
        text = " ".join(tokens)
        # Small cleanup for punctuation spacing.
        for p in [".", ",", "!", "?", ":", ";"]:
            text = text.replace(f" {p}", p)
        text = text.replace(" n't", "n't")
        text = text.replace(" 'm", "'m").replace(" 're", "'re").replace(" 's", "'s").replace(" 've", "'ve").replace(" 'll", "'ll")
        return text.strip()

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """Translate a German sentence to English using greedy decoding."""
        self.eval()
        device = next(self.parameters()).device
        src_tokens = self._tokenize(src_sentence, lang="de")
        src_indices = torch.tensor([self._numericalize(src_tokens, self.src_vocab)], dtype=torch.long, device=device)
        src_mask = make_src_mask(src_indices, pad_idx=self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src_indices, src_mask)
            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)
            max_len = min(max(src_indices.size(1) + 8, 8), 15)
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                out = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == self.eos_idx:
                    break

        tokens = ys[0].tolist()
        words = []
        for idx in tokens:
            if idx in (self.sos_idx, self.eos_idx, self.pad_idx):
                continue
            words.append(self.tgt_vocab.lookup_token(idx))
        return self._detokenize(words)
