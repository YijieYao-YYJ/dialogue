#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
from pathlib import Path
from collections import Counter
from typing import Any, List, Dict

# -------------------- Config --------------------
QUERIES_PATH  = Path("../results/quiries/queries_test.json")
ENTITIES_PATH = Path("../results/entities/entities.json")
OUT_PATH      = Path("../results/bm25/bm25_constraints_results.json")

ALLOWED_DOMAINS = {"restaurant", "hotel", "attraction"}
TOPK = 10
K1   = 1.2
B    = 0.75

# -------------------- Tokenize --------------------
_token_re = re.compile(r"[a-z0-9]+")
def tokenize(text: str):
    if not text:
        return []
    text = text.lower()
    return _token_re.findall(text)


# -------------------- constraints -> text --------------------
EMPTY_TOKENS = {
    "",
    "not mentioned",
    "dontcare",
    "do n't care",
    "dont care",
    "none",
    "don't care",
}

def is_filled(v: Any) -> bool:
    if v is None:
        return False
    if not isinstance(v, str):
        return True
    return v.strip().lower() not in EMPTY_TOKENS


def constraints_to_text(q: Dict[str, Any]) -> str:

    cons_list = q.get("constraints") or []
    parts: List[str] = []

    for c in cons_list:
        if not isinstance(c, dict):
            continue
        dom = (c.get("domain") or "").strip().lower()
        if dom not in ALLOWED_DOMAINS:
            continue

        kv_list = c.get("kv") or []
        clean_kv = []
        for k, v in kv_list:
            if not is_filled(v):
                continue
            k_str = str(k).strip().lower()
            v_str = str(v).strip().lower()
            if not v_str:
                continue
            clean_kv.append((k_str, v_str))

        if not clean_kv:
            continue

        parts.append(dom)
        for k, v in clean_kv:
            parts.append(k)
            parts.append(v)

    return " ".join(parts)


# -------------------- Load entities --------------------
def load_entities(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data if (e.get("domain") in ALLOWED_DOMAINS)]


# -------------------- BM25 Index --------------------
class BM25Index:
    def __init__(self, docs, get_text):

        self.docs = docs
        self.get_text = get_text

        self.doc_tokens = []
        self.doc_tf = []
        self.df = Counter()
        self.N = len(docs)

        total_len = 0
        for doc in docs:
            toks = tokenize(get_text(doc))
            self.doc_tokens.append(toks)
            total_len += len(toks)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            for w in tf.keys():
                self.df[w] += 1

        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0
        self.idf = {
            w: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            for w, n in self.df.items()
        }
        self.doc_len = [len(toks) for toks in self.doc_tokens]

    def score(self, q_tokens, idx):
        tf = self.doc_tf[idx]
        dl = self.doc_len[idx]
        score = 0.0
        for w in q_tokens:
            if w not in tf:
                continue
            f = tf[w]
            idf = self.idf.get(w, 0.0)
            denom = f + K1 * (1 - B + B * (dl / (self.avgdl or 1.0)))
            score += idf * (f * (K1 + 1)) / (denom or 1e-9)
        return score

    def query(self, q_text: str, allowed_domains=None, topk: int = 5):
        q_tokens = tokenize(q_text)
        if not q_tokens:
            return []

        scores = []
        for i, doc in enumerate(self.docs):
            if allowed_domains is not None and doc.get("domain") not in allowed_domains:
                continue
            s = self.score(q_tokens, i)
            # if s > 0:
            #     scores.append((s, i))
            scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:topk]


# -------------------- Helpers --------------------
def pack_top_items(entities, hits):
    out = []
    for score, idx in hits:
        doc = entities[idx]
        attrs = doc.get("attrs") or {}
        out.append({
            "doc_id": doc.get("doc_id"),
            "domain": doc.get("domain"),
            "score": round(float(score), 6),
            "name": attrs.get("name"),
            "area": attrs.get("area"),
            "pricerange": attrs.get("pricerange"),
        })
    return out


def _unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -------------------- Main --------------------
def main():
    if not ENTITIES_PATH.exists():
        raise FileNotFoundError(f"entities.json 不存在: {ENTITIES_PATH}")
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"queries_dev.json 不存在: {QUERIES_PATH}")

    entities = load_entities(ENTITIES_PATH)
    if not entities:
        print("No entities loaded. Check entities.json and ALLOWED_DOMAINS.")
        OUT_PATH.write_text("[]", encoding="utf-8")
        return

    bm25 = BM25Index(entities, get_text=lambda d: d.get("text", ""))
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    results = []
    for q in queries:
        cons_list = q.get("constraints") or []
        if not cons_list:
            continue

        # 域来源：constraints 中的 domain
        domains = [c.get("domain") for c in cons_list if isinstance(c, dict)]
        domains = [d for d in domains if d in ALLOWED_DOMAINS]
        domains = _unique_keep_order(domains)
        if not domains:
            continue

        # query 文本来自 dialogue state
        q_text = constraints_to_text(q)
        if not q_text.strip():
            continue

        per_domain = []
        for dom in domains:
            hits = bm25.query(q_text, allowed_domains={dom}, topk=TOPK)
            per_domain.append({
                "domain": dom,
                "topk": pack_top_items(entities, hits)
            })

        results.append({
            "dialogue_id": q.get("dialogue_id"),
            "turn": q.get("turn"),
            "domains": domains,
            "results": per_domain
        })

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BM25(constraints) 完成检索，写出 {len(results)} 条结果 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
