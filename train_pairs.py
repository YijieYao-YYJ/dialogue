#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 MultiWOZ 抽取好的 queries_train.json + entities.json + dbquery.Database
构造 RNN 双塔训练用的样本对

输出格式 (train_pairs.json):

[
  {
    "dialogue_id": "SNG01856.json",
    "turn": 7,
    "domain": "restaurant",
    "query": "...query_text...",
    "pos_text": "...正样本实体 text...",
    "pos_name": "gold restaurant name",
    "neg_texts": ["...负样本1 text...", "..."],
    "neg_names": ["neg name 1", "neg name 2", ...]
  },
  ...
]

"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set


# -------------------- Config --------------------

QUERIES_PATH  = Path("results/quiries/queries_train.json")
ENTITIES_PATH = Path("results/entities/entities.json")
OUT_PATH      = Path("gru/dialogue_state/train_pairs.json")


ALLOWED_DOMAINS: Set[str] = {"restaurant", "hotel", "attraction"}

EMPTY_TOKENS = {"", "not mentioned", "dontcare", "do n't care", "dont care", "none", "don't care"}

# 每个正样本配多少个负样本
NEG_PER_POS = 5

RNG_SEED = 2025


# -------------------- Utils --------------------


def is_filled(v: Any) -> bool:
    if v is None:
        return False
    if not isinstance(v, str):
        return True
    return v.strip().lower() not in EMPTY_TOKENS


def load_raw_queries(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_constraints(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cons_list = raw or []
    out = []
    for c in cons_list:
        dom = c.get("domain")
        if dom not in ALLOWED_DOMAINS:
            continue
        kv_raw = c.get("kv") or []
        kv = [(k, v) for (k, v) in kv_raw if is_filled(v)]
        if not kv:
            continue
        out.append({"domain": dom, "kv": kv})
    return out

def constraints_to_text(cons_list: List[Dict[str, Any]]) -> str:
    """
    把一个 USER 轮里的 dialogue state(constraints) 转成一段查询字符串，
    比如: restaurant pricerange cheap area centre food chinese hotel stars 4 area north ...
    """
    parts: List[str] = []

    for c in cons_list:
        if not isinstance(c, dict):
            continue
        dom = (c.get("domain") or "").strip().lower()
        if dom not in ALLOWED_DOMAINS:
            continue

        kv_list = c.get("kv") or []
        clean_kv: List[Tuple[str, str]] = []
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


def load_entities(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_entity_index(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    index: Dict[str, Dict[str, Dict[str, Any]]] = {d: {} for d in ALLOWED_DOMAINS}
    for ent in entities:
        dom = ent.get("domain")
        if dom not in ALLOWED_DOMAINS:
            continue
        attrs = ent.get("attrs") or {}
        name = (attrs.get("name") or "").strip().lower()
        if not name:
            continue

        if name not in index[dom]:
            index[dom][name] = ent
    return index


# -------------------- Core --------------------


def build_train_pairs() -> List[Dict[str, Any]]:
    from dbquery import Database

    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"找不到 queries_train.json: {QUERIES_PATH}")
    if not ENTITIES_PATH.exists():
        raise FileNotFoundError(f"找不到 entities.json: {ENTITIES_PATH}")

    random.seed(RNG_SEED)

    raw_queries = load_raw_queries(QUERIES_PATH)
    entities = load_entities(ENTITIES_PATH)
    ent_index = build_entity_index(entities)


    all_ents_by_dom: Dict[str, List[Tuple[str, str]]] = {d: [] for d in ALLOWED_DOMAINS}
    for dom in ALLOWED_DOMAINS:
        for name_lc, ent in ent_index[dom].items():
            all_ents_by_dom[dom].append((name_lc, ent.get("text") or ""))

    db = Database()

    pairs: List[Dict[str, Any]] = []
    total_queries = 0
    used_queries = 0
    skipped_no_query_text = 0
    skipped_no_constraints = 0
    skipped_no_gold = 0
    skipped_no_negs = 0

    for it in raw_queries:
        total_queries += 1

        # q_text = (it.get("query_text") or "").strip()
        # if not q_text:
        #     skipped_no_query_text += 1
        #     continue

        did = it.get("dialogue_id")
        turn = int(it.get("turn", 0))
        cons_norm = normalize_constraints(it.get("constraints") or [])
        #dialogue state
        q_text = constraints_to_text(cons_norm)
        if not q_text.strip():
            skipped_no_query_text += 1
            continue

        if not cons_norm:
            skipped_no_constraints += 1
            continue

        for c in cons_norm:
            dom = c["domain"]
            kv = c["kv"]
            gold_records = db.query(dom, kv)
            gold_names = {
                (rec.get("name") or "").strip().lower()
                for rec in gold_records
                if isinstance(rec.get("name"), str)
            }
            gold_names.discard("")

            if not gold_names:
                skipped_no_gold += 1
                continue

            pos_entities: List[Tuple[str, str]] = []  # (name_lc, text)
            for nm in gold_names:
                ent = ent_index.get(dom, {}).get(nm)
                if ent:
                    pos_entities.append((nm, ent.get("text") or ""))

            if not pos_entities:
                skipped_no_gold += 1
                continue

            # 为该域采负样本：从 all_ents_by_dom[dom] 中剔除 gold name
            cand_negs = [
                (nm, txt)
                for (nm, txt) in all_ents_by_dom.get(dom, [])
                if nm not in gold_names and txt.strip()
            ]
            if len(cand_negs) < NEG_PER_POS:
                # 负样本太少,跳过
                skipped_no_negs += 1
                continue

            for pos_name_lc, pos_text in pos_entities:
                # 随机挑 NEG_PER_POS 个负样本
                neg_samples = random.sample(cand_negs, NEG_PER_POS)
                neg_names = [nm for (nm, _) in neg_samples]
                neg_texts = [txt for (_, txt) in neg_samples]

                pair = {
                    "dialogue_id": did,
                    "turn": turn,
                    "domain": dom,
                    "query": q_text,
                    "pos_text": pos_text,
                    "pos_name": pos_name_lc,
                    "neg_texts": neg_texts,
                    "neg_names": neg_names,
                }
                pairs.append(pair)
                used_queries += 1

    print(f"总 query 条数: {total_queries}")
    print(f"生成训练样本数: {len(pairs)}")
    print(f"  - 跳过(无 query_text): {skipped_no_query_text}")
    print(f"  - 跳过(无有效 constraints): {skipped_no_constraints}")
    print(f"  - 跳过(DB 中无 gold): {skipped_no_gold}")
    print(f"  - 跳过(负样本不足): {skipped_no_negs}")

    return pairs


def main():
    pairs = build_train_pairs()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出训练数据 -> {OUT_PATH}，共 {len(pairs)} 条样本。")


if __name__ == "__main__":
    main()
