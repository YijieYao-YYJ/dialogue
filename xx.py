#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Set

QUERIES_PATH      = Path("queries_dev.json")
BM25_RESULTS_PATH = Path("results/bm25/bm25_results.json")

ALLOWED_DOMAINS: Set[str] = {"restaurant", "hotel", "attraction"}
EMPTY_TOKENS = {"", "not mentioned", "dontcare", "do n't care", "dont care", "none", "don't care"}

def is_filled(v: Any) -> bool:
    if v is None:
        return False
    if not isinstance(v, str):
        return True
    return v.strip().lower() not in EMPTY_TOKENS

def load_queries(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for it in data:
        did = it.get("dialogue_id")
        turn = int(it.get("turn", 0))
        cons_list = it.get("constraints") or []
        # 仅保留允许域 + 非空kv 的约束（与评测一致）
        cons_list = [
            {"domain": c.get("domain"),
             "kv": [(k, v) for k, v in (c.get("kv") or []) if is_filled(v)]}
            for c in cons_list
            if c.get("domain") in ALLOWED_DOMAINS and (c.get("kv") or [])
        ]
        out.append({"dialogue_id": did, "turn": turn, "constraints": cons_list})
    return out

def load_bm25(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    pred = {}
    for item in data:
        did = item.get("dialogue_id")
        turn = int(item.get("turn", 0))
        for block in item.get("results", []):
            dom = block.get("domain")
            if dom not in ALLOWED_DOMAINS:
                continue
            names = []
            for hit in block.get("topk", []):
                nm = (hit.get("name") or "").strip().lower()
                if nm:
                    names.append(nm)
            # 注意：即使 names 为空，也记录键，表示该样本“有预测但为空”
            pred[(did, turn, dom)] = names
    return pred

def main():
    queries = load_queries(QUERIES_PATH)
    pred_index = load_bm25(BM25_RESULTS_PATH)

    missing_dids = set()

    for item in queries:
        did, turn = item["dialogue_id"], int(item["turn"])
        for c in item["constraints"]:
            dom = c["domain"]
            if dom not in ALLOWED_DOMAINS:
                continue
            kv = c.get("kv") or []
            if not kv:
                continue

            pred_names = pred_index.get((did, turn, dom), None)
            # 判定“该约束实例预测为空”：
            # 1) 根本没有这个键（None）或
            # 2) 这个键存在但列表为空（len==0）
            if not pred_names:
                missing_dids.add(did)

    # 输出（只要 dialogue_id，去重）
    print("Missing-pred dialogue_id count:", len(missing_dids))
    for did in sorted(missing_dids):
        print(did)

if __name__ == "__main__":
    main()







