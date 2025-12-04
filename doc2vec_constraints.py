#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# -------------------- Config --------------------
QUERIES_PATH   = Path("results/quiries/queries_dev.json")
ENTITIES_PATH  = Path("results/entities/entities.json")
MODEL_PATH     = Path("doc2vec_constraints.bin")
OUT_PATH       = Path("results/doc2vec/doc2vec_constraints_results.json")

ALLOWED_DOMAINS = {"restaurant", "hotel", "attraction"}
TOPK = 5

VECTOR_SIZE = 150
WINDOW      = 3
MIN_COUNT   = 1
NEGATIVE    = 10
EPOCHS      = 100
SEED        = 2025
WORKERS     = max(1, os.cpu_count() - 1)
SAMPLE      = 1e-3


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


# -------------------- Tokenize --------------------
_token_re = re.compile(r"[a-z0-9]+")
def tokenize(text: str):
    if not text:
        return []
    text = text.lower()
    return _token_re.findall(text)


# -------------------- IO --------------------
def load_entities(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ents = [e for e in data if (e.get("domain") in ALLOWED_DOMAINS)]
    return ents


def load_queries(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------- Dialogue state -> tokens --------------------
def constraints_to_tokens(q: Dict[str, Any]) -> List[str]:
    """
    把一个 query 里的 dialogue state (constraints)
    转成一串 tokens，用于 doc2vec 的输入。

    """
    cons_list = q.get("constraints") or []
    parts: List[str] = []

    for c in cons_list:
        if not isinstance(c, dict):
            continue
        dom = (c.get("domain") or "").strip().lower()
        if dom not in ALLOWED_DOMAINS:
            continue

        kv_list = c.get("kv") or []
        # 过滤掉空值 / dontcare / none 等
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

    if not parts:
        return []

    text = " ".join(parts)
    return tokenize(text)


# -------------------- Train --------------------
def build_tagged_docs(entities: List[Dict[str, Any]]) -> List[TaggedDocument]:

    tagged: List[TaggedDocument] = []
    for ent in entities:
        doc_id = ent.get("doc_id") or ""
        text   = ent.get("text") or ""
        tokens = tokenize(text)
        if not doc_id or not tokens:
            continue
        tagged.append(TaggedDocument(words=tokens, tags=[doc_id]))
    return tagged


def build_query_tagged_docs(queries: List[Dict[str, Any]]) -> List[TaggedDocument]:

    tagged: List[TaggedDocument] = []
    for q in queries:
        tokens = constraints_to_tokens(q)
        if not tokens:
            continue

        qtag = f"__Q__{q.get('dialogue_id','?')}__{q.get('turn','?')}"
        tagged.append(TaggedDocument(words=tokens, tags=[qtag]))
    return tagged


def train_or_load_model(entities, queries) -> Doc2Vec:
    if MODEL_PATH.exists():
        return Doc2Vec.load(str(MODEL_PATH))

    # 1) 实体语料
    tagged_entities = build_tagged_docs(entities)

    # 2) 对话状态语料（constraints）
    tagged_queries = build_query_tagged_docs(queries)

    # 3) 合并训练语料（实体 + 对话状态）
    tagged_all = tagged_queries + tagged_entities
    if not tagged_all:
        raise RuntimeError("No training data found. Check entities.json and queries_dev.json.")

    # 4) 训练 Doc2Vec（PV-DBOW）
    model = Doc2Vec(
        dm=0,                 # PV-DBOW
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        # min_count=MIN_COUNT,
        negative=NEGATIVE,
        seed=SEED,
        workers=WORKERS,
        dbow_words=1,
        hs=0,
        epochs=EPOCHS,
        # sample=SAMPLE,
    )

    model.build_vocab(tagged_all)
    model.train(tagged_all, total_examples=len(tagged_all), epochs=model.epochs)

    model.save(str(MODEL_PATH))
    return model


# -------------------- Search --------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / denom


def cosine_topk(query_vec: np.ndarray, cand_mat: np.ndarray, k: int) -> List[int]:
    #  L2 归一化后，点积即余弦
    scores = cand_mat @ query_vec
    if len(scores) <= k:
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, kth=k-1)[:k]
    idx_sorted = idx[np.argsort(-scores[idx])]
    return list(idx_sorted)


def build_domain_index(model: Doc2Vec, entities):

    by_dom = {d: {"doc_ids": [], "attrs": [], "mat": None} for d in ALLOWED_DOMAINS}
    vecs   = {d: [] for d in ALLOWED_DOMAINS}

    for ent in entities:
        dom = ent.get("domain")
        if dom not in ALLOWED_DOMAINS:
            continue
        doc_id = ent.get("doc_id")
        if doc_id not in model.dv:
            continue

        by_dom[dom]["doc_ids"].append(doc_id)
        by_dom[dom]["attrs"].append(ent.get("attrs") or {})
        vecs[dom].append(model.dv[doc_id])

    for dom in ALLOWED_DOMAINS:
        if vecs[dom]:
            mat = np.vstack(vecs[dom]).astype(np.float32)
            by_dom[dom]["mat"] = l2_normalize(mat)
        else:
            by_dom[dom]["mat"] = np.zeros((0, model.vector_size), dtype=np.float32)

    return by_dom


def pack_top_items(doc_ids, attrs_list, order, scores):
    out = []
    for rank, idx in enumerate(order):
        doc_id = doc_ids[idx]
        attrs  = attrs_list[idx]
        out.append({
            "doc_id": doc_id,
            "domain": None,
            "score": float(scores[idx]),
            "name": (attrs or {}).get("name"),
            "area": (attrs or {}).get("area"),
            "pricerange": (attrs or {}).get("pricerange"),
        })
    return out


def search_and_write(model: Doc2Vec, entities, queries):
    # 构建按域的实体库与向量矩阵
    dom_index = build_domain_index(model, entities)

    results = []
    for q in queries:
        cons_list = q.get("constraints") or []
        if not cons_list:
            continue

        # 域以 constraints 为准
        domains = [c.get("domain") for c in cons_list if isinstance(c, dict)]
        domains = [d for d in domains if d in ALLOWED_DOMAINS]
        if not domains:
            continue

        # ------------- ：query 用 dialogue state -------------
        q_tokens = constraints_to_tokens(q)
        if not q_tokens:
            continue

        # 推断 query 向量并 L2 归一化
        q_vec = model.infer_vector(q_tokens, epochs=100)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        per_domain = []
        for dom in domains:
            doc_ids = dom_index[dom]["doc_ids"]
            attrs   = dom_index[dom]["attrs"]
            mat     = dom_index[dom]["mat"]   # shape: [n_doc, dim]

            if mat.shape[0] == 0:
                continue

            # 计算余弦相似（点积）
            scores = mat @ q_vec
            order  = cosine_topk(q_vec, mat, TOPK)

            packed = pack_top_items(doc_ids, attrs, order, scores)
            # 填回 domain 字段
            for it in packed:
                it["domain"] = dom

            per_domain.append({
                "domain": dom,
                "topk": packed
            })

        results.append({
            "dialogue_id": q.get("dialogue_id"),
            "turn": q.get("turn"),
            "domains": domains,
            "results": per_domain
        })

    OUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ Doc2Vec(PV-DBOW, constraints) 完成检索，写出 {len(results)} 条结果 -> {OUT_PATH}")


def main():
    if not ENTITIES_PATH.exists():
        raise FileNotFoundError("entities.json 不存在。")
    if not QUERIES_PATH.exists():
        raise FileNotFoundError("queries_dev.json 不存在。")

    entities = load_entities(ENTITIES_PATH)
    queries  = load_queries(QUERIES_PATH)
    if not entities:
        raise RuntimeError("No valid entities found.")
    if not queries:
        raise RuntimeError("No valid queries found.")

    model = train_or_load_model(entities, queries)
    search_and_write(model, entities, queries)


if __name__ == "__main__":
    main()
