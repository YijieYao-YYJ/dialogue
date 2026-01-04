#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN 双塔检索脚本（BiGRU + 向量归一化）

输入：
  - gru.pt
  - gru_vocab.json
  - results/entities/entities.json
  - results/quiries/queries_dev.json

输出：
  - results/gru/gru_results.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dbquery import Database

EMPTY_TOKENS = {"", "not mentioned", "dontcare", "do n't care", "dont care", "none", "don't care"}

VOCAB_PATH        = Path("gru_vocab.json")
MODEL_PATH        = Path("gru.pt")
ENTITIES_PATH     = Path("../../results/entities/entities.json")
QUERIES_DEV_PATH  = Path("../../results/quiries/queries_test_user.json")
OUT_PATH          = Path("../../results/gru/dialogue_only_user/gru_results.json")

ALLOWED_DOMAINS = {"restaurant", "hotel", "attraction"}


def simple_tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, token2id: Dict[str, int], id2token: List[str]):
        self.token2id = token2id
        self.id2token = id2token

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2id[self.UNK]
        return [self.token2id.get(t, unk) for t in tokens]

    def pad(self, seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        PAD_ID = self.token2id[self.PAD]
        padded = []
        for s in seqs:
            if len(s) < max_len:
                s = s + [PAD_ID] * (max_len - len(s))
            padded.append(s)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


class BiGRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
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

    def forward(self, input_ids, lengths):
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        input_sorted = input_ids[sorted_idx]

        emb = self.embedding(input_sorted)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, h_n = self.gru(packed)

        fw = h_n[-2]  # forward last layer
        bw = h_n[-1]  # backward last layer
        out_sorted = torch.cat([fw, bw], dim=-1)
        out_sorted = self.dropout(out_sorted)

        _, inv_idx = sorted_idx.sort()
        return out_sorted[inv_idx]


class TwinTowerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = BiGRUEncoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def encode(self, ids, lens):
        return self.encoder(ids, lens)


def load_entities() -> Dict[str, List[Dict[str, Any]]]:
    ents = json.loads(ENTITIES_PATH.read_text(encoding="utf-8"))
    index = {d: [] for d in ALLOWED_DOMAINS}
    for e in ents:
        d = e.get("domain")
        if d in ALLOWED_DOMAINS:
            index[d].append(e)
    return index


def encode_entities(model: TwinTowerModel, vocab: Vocab,
                    ent_index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    vecs: Dict[str, torch.Tensor] = {}

    for dom, es in ent_index.items():
        if not es:
            vecs[dom] = torch.empty(0, model.encoder.gru.hidden_size * 2, device=device)
            continue
        texts = [e["text"] for e in es]
        toks = [simple_tokenize(t) for t in texts]
        ids  = [vocab.encode(ts) for ts in toks]

        ids_tensor, lens_tensor = vocab.pad(ids)
        ids_tensor = ids_tensor.to(device)
        lens_tensor = lens_tensor.to(device)

        with torch.no_grad():
            emb = model.encode(ids_tensor, lens_tensor)
            emb = F.normalize(emb, p=2, dim=-1)
        vecs[dom] = emb
    return vecs


def retrieve():
    vocab_json = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab = Vocab(vocab_json["token2id"], vocab_json["id2token"])
    cfg = vocab_json["config"]

    model = TwinTowerModel(
        vocab_size=len(vocab.id2token),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    ent_index = load_entities()
    ent_vecs  = encode_entities(model, vocab, ent_index)

    queries = json.loads(QUERIES_DEV_PATH.read_text(encoding="utf-8"))

    results = []
    db = Database()

    for q in queries:
        q_text = q["query_text"]

        filtered = []
        for c in q.get("constraints", []):
            dom = c["domain"]
            if dom not in ALLOWED_DOMAINS:
                continue
            kv = [
                (k, v)
                for (k, v) in c["kv"]
                if isinstance(v, str) and v.strip().lower() not in EMPTY_TOKENS
            ]
            if kv:
                filtered.append({"domain": dom, "kv": kv})

        item = {
            "dialogue_id": q["dialogue_id"],
            "turn": q["turn"],
            "domains": [],
            "results": [],
        }

        if not filtered:
            results.append(item)
            continue

        toks = simple_tokenize(q_text)
        ids  = vocab.encode(toks)
        ids_tensor, lens_tensor = vocab.pad([ids])
        ids_tensor = ids_tensor.to(device)
        lens_tensor = lens_tensor.to(device)

        with torch.no_grad():
            qvec = model.encode(ids_tensor, lens_tensor)[0]
            qvec = F.normalize(qvec, p=2, dim=-1)

        for c in filtered:
            dom = c["domain"]
            item["domains"].append(dom)

            ev = ent_vecs[dom]
            if ev.numel() == 0:
                item["results"].append({"domain": dom, "topk": []})
                continue

            scores = torch.sum(ev * qvec.unsqueeze(0), dim=-1)
            TOP_K = min(20, ev.size(0))
            topk = torch.topk(scores, k=TOP_K)

            ents = ent_index[dom]
            out_items = []
            for idx, s in zip(topk.indices.tolist(), topk.values.tolist()):
                e = ents[idx]
                out_items.append({
                    "doc_id": e["doc_id"],
                    "domain": dom,
                    "score": float(s),
                    **e["attrs"],
                })

            item["results"].append({"domain": dom, "topk": out_items})

        results.append(item)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("检索完成，写出", OUT_PATH)


if __name__ == "__main__":
    retrieve()