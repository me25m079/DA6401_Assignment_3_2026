from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional


def _blank_tokenizer(lang: str):
    try:
        import spacy
        return spacy.blank(lang)
    except Exception:
        return None


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]

    @classmethod
    def build(cls, tokens: Iterable[str], min_freq: int = 1) -> "Vocab":
        special = ["<unk>", "<pad>", "<sos>", "<eos>"]
        counter = Counter(tokens)
        itos = list(special)
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in special:
                itos.append(tok)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(itos=itos, stoi=stoi)

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def lookup_token(self, index: int) -> str:
        if 0 <= index < len(self.itos):
            return self.itos[index]
        return "<unk>"


class Multi30kDataset:
    def __init__(self, split: str = 'train'):
        self.split = split
        self.de_tokenizer = _blank_tokenizer("de")
        self.en_tokenizer = _blank_tokenizer("en")

        self.raw_data = []
        try:
            from datasets import load_dataset
            ds = load_dataset("bentrevett/multi30k")
            split_name = "validation" if split in {"val", "valid", "validation"} else split
            if split_name not in ds:
                split_name = "train"
            self.raw_data = ds[split_name]
        except Exception:
            # Offline fallback: minimal toy data so the class remains usable.
            self.raw_data = [
                {"de": "ein mann steht auf einer straße .", "en": "a man stands on a street ."},
                {"de": "eine frau sitzt auf einer bank .", "en": "a woman sits on a bench ."},
                {"de": "ein kind spielt mit einem ball .", "en": "a child plays with a ball ."},
            ]

        self.src_vocab: Optional[Vocab] = None
        self.tgt_vocab: Optional[Vocab] = None

    def _tokenize(self, text: str, lang: str) -> list[str]:
        tokenizer = self.de_tokenizer if lang == "de" else self.en_tokenizer
        if tokenizer is None:
            return text.strip().split()
        try:
            return [tok.text for tok in tokenizer.tokenizer(text.strip())]
        except Exception:
            return text.strip().split()

    def build_vocab(self):
        src_tokens = []
        tgt_tokens = []
        for sample in self.raw_data:
            src_tokens.extend(self._tokenize(sample["de"], "de"))
            tgt_tokens.extend(self._tokenize(sample["en"], "en"))
        self.src_vocab = Vocab.build(src_tokens)
        self.tgt_vocab = Vocab.build(tgt_tokens)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()

        processed = []
        for sample in self.raw_data:
            src_tokens = ["<sos>"] + self._tokenize(sample["de"], "de") + ["<eos>"]
            tgt_tokens = ["<sos>"] + self._tokenize(sample["en"], "en") + ["<eos>"]
            src_ids = [self.src_vocab.lookup_index(tok) for tok in src_tokens]
            tgt_ids = [self.tgt_vocab.lookup_index(tok) for tok in tgt_tokens]
            processed.append((src_ids, tgt_ids))
        return processed

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        return sample["de"], sample["en"]
