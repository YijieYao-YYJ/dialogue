#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN（BiGRU + in-batch softmax / InfoNCE）

输入：
  - results/rnn/train_pairs.json

输出：
  - rnn_vocab.json
  - rnn_gru.pt
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------- 路径与超参数 --------------------

TRAIN_PAIRS_PATH = Path("results/rnn/train_pairs.json")
VOCAB_PATH       = Path("rnn_vocab.json")
MODEL_PATH       = Path("rnn_gru.pt")

EMBED_DIM   = 128
HIDDEN_DIM  = 128
NUM_LAYERS  = 1
DROPOUT     = 0.1

BATCH_SIZE  = 64
NUM_EPOCHS  = 5
LR          = 1e-3
TEMPERATURE = 0.05

MIN_FREQ    = 1


# -------------------- tokenizer & Vocab --------------------

def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.lower().strip().split()


class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self.add_token(self.PAD)
        self.add_token(self.UNK)

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.id2token)
            self.token2id[token] = idx
            self.id2token.append(token)
        return self.token2id[token]

    def build_from_corpus(self, corpus: List[List[str]], min_freq: int = 1) -> None:
        from collections import Counter
        counter = Counter()
        for toks in corpus:
            counter.update(toks)
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.token2id:
                self.add_token(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2id[self.UNK]
        return [self.token2id.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.id2token)


# -------------------- Dataset & collate --------------------

class PairDataset(Dataset):

    def __init__(self, pairs: List[Dict[str, Any]], vocab: Vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        it = self.pairs[idx]
        q_text: str   = it["query"]
        pos_text: str = it["pos_text"]

        q_ids   = self.vocab.encode(simple_tokenize(q_text))
        pos_ids = self.vocab.encode(simple_tokenize(pos_text))

        return {
            "q_ids": q_ids,
            "pos_ids": pos_ids,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    def pad_sequences(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        padded = []
        for s in seqs:
            if len(s) < max_len:
                s = s + [pad_id] * (max_len - len(s))
            padded.append(s)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    pad_id = 0

    q_ids_list   = [b["q_ids"] for b in batch]
    pos_ids_list = [b["pos_ids"] for b in batch]

    q_ids, q_lens       = pad_sequences(q_ids_list, pad_id)
    pos_ids, pos_lens   = pad_sequences(pos_ids_list, pad_id)

    return {
        "q_ids": q_ids,
        "q_lens": q_lens,
        "pos_ids": pos_ids,
        "pos_lens": pos_lens,
    }


# -------------------- BiGRU Encoder & Model --------------------

class BiGRUEncoder(nn.Module):


    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        input_sorted = input_ids[sorted_idx]

        emb = self.embedding(input_sorted)  # (B,L,E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, h_n = self.gru(packed)
        # h_n: (num_layers * num_directions, B, H)
        num_directions = 2
        fw = h_n[-num_directions]   # (B,H)
        bw = h_n[-1]                # (B,H)
        enc_sorted = torch.cat([fw, bw], dim=-1)  # (B,2H)
        enc_sorted = self.dropout(enc_sorted)

        _, inv_idx = sorted_idx.sort()
        return enc_sorted[inv_idx]


class TwinTowerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = BiGRUEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids, lengths)


# -------------------- 训练 --------------------

def load_train_pairs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"train_pairs.json 不存在：{path}")
    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        raise ValueError("train_pairs.json 顶层必须是 list")
    return pairs


def build_vocab_from_pairs(pairs: List[Dict[str, Any]], min_freq: int) -> Vocab:
    corpus: List[List[str]] = []
    for it in pairs:
        q_tokens   = simple_tokenize(it["query"])
        pos_tokens = simple_tokenize(it["pos_text"])
        neg_tokens_all = []
        for t in it.get("neg_texts", []):
            neg_tokens_all.extend(simple_tokenize(t))
        corpus.append(q_tokens)
        corpus.append(pos_tokens)
        if neg_tokens_all:
            corpus.append(neg_tokens_all)

    vocab = Vocab()
    vocab.build_from_corpus(corpus, min_freq=min_freq)
    return vocab


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    pairs = load_train_pairs(TRAIN_PAIRS_PATH)
    print("加载 train_pairs:", len(pairs), "条样本。")

    vocab = build_vocab_from_pairs(pairs, MIN_FREQ)
    print("词表大小:", len(vocab))

    dataset = PairDataset(pairs, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = TwinTowerModel(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            q_ids    = batch["q_ids"].to(device)
            q_lens   = batch["q_lens"].to(device)
            pos_ids  = batch["pos_ids"].to(device)
            pos_lens = batch["pos_lens"].to(device)

            optimizer.zero_grad()

            q_vec   = model.encode(q_ids,  q_lens)
            pos_vec = model.encode(pos_ids, pos_lens)

            # L2 归一化
            q_vec   = F.normalize(q_vec,   p=2, dim=-1)
            pos_vec = F.normalize(pos_vec, p=2, dim=-1)

            # 相似度矩阵 (B,B)
            scores = torch.matmul(q_vec, pos_vec.t())
            scores = scores / TEMPERATURE

            batch_size = scores.size(0)
            labels = torch.arange(batch_size, device=device)

            # 对称 InfoNCE
            loss_q2p = F.cross_entropy(scores, labels)
            loss_p2q = F.cross_entropy(scores.t(), labels)
            loss = (loss_q2p + loss_p2q) / 2.0

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] loss = {avg_loss:.4f}")

    print("保存模型和词表...")
    torch.save(model.state_dict(), MODEL_PATH)

    vocab_payload = {
        "token2id": vocab.token2id,
        "id2token": vocab.id2token,
        "config": {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "min_freq": MIN_FREQ,
            "temperature": TEMPERATURE,
        },
    }
    VOCAB_PATH.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(" 训练完成，模型保存到", MODEL_PATH)
    print(" 词表保存到", VOCAB_PATH)


if __name__ == "__main__":
    train()


