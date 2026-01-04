import json
from pathlib import Path
from typing import Any, Set


QUERIES_PATH = Path("results/quiries/queries_test.json")
RESULTS_PATH = Path("results/doc2vec/doc2vec_constraints_results.json")

# 参与评测的域
ALLOWED_DOMAINS: Set[str] = {"restaurant", "hotel", "attraction"}

# 评测的 K 值
K_LIST = [1, 3, 5, 10]


EMPTY_TOKENS = {"", "not mentioned", "dontcare", "do n't care", "dont care", "none", "don't care"}


def is_filled(v: Any) -> bool:
    if v is None:
        return False
    if not isinstance(v, str):
        return True
    return v.strip().lower() not in EMPTY_TOKENS


def load_queries(path: Path):
    """读取：仅保留允许域、非空 kv 的 constraints。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for it in data:
        did = it.get("dialogue_id")
        turn = int(it.get("turn", 0))
        cons_list = it.get("constraints") or []
        cons_list = [
            {
                "domain": c.get("domain"),
                "kv": [(k, v) for k, v in (c.get("kv") or []) if is_filled(v)]
            }
            for c in cons_list
            if c.get("domain") in ALLOWED_DOMAINS and (c.get("kv") or [])
        ]
        out.append({"dialogue_id": did, "turn": turn, "constraints": cons_list})
    return out


def load_bm25(path: Path):
    """读取检索结果（doc2vec_results.json / bm25_results.json），
    建立 (dialogue_id, turn, domain) -> [name1, name2, ...] 的索引。"""
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
            pred[(did, turn, dom)] = names
    return pred


def get_database():
    try:
        from dbquery import Database
        return Database()
    except Exception as e:
        raise RuntimeError(f"Cannot import/use dbquery.Database: {e}")


def prf(tp: int, p: int, g: int):
    """计算 Precision / Recall / F1。"""
    prec = tp / p if p else 0.0
    rec  = tp / g if g else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def reciprocal_rank_at_k(pred_names, gold_names, k: int) -> float:

    topk = pred_names[:k]
    for idx, name in enumerate(topk):
        if name in gold_names:
            return 1.0 / (idx + 1)
    return 0.0


def evaluate():
    queries = load_queries(QUERIES_PATH)
    pred_index = load_bm25(RESULTS_PATH)
    db = get_database()

    overall = {K: {"tp": 0, "p": 0, "g": 0, "samples": 0} for K in K_LIST}
    bydom   = {
        d: {K: {"tp": 0, "p": 0, "g": 0, "samples": 0} for K in K_LIST}
        for d in ALLOWED_DOMAINS
    }


    overall_rr_sum = {K: 0.0 for K in K_LIST}
    bydom_rr_sum   = {
        d: {K: 0.0 for K in K_LIST}
        for d in ALLOWED_DOMAINS
    }

    for item in queries:
        did, turn = item["dialogue_id"], int(item["turn"])
        for c in item["constraints"]:
            dom = c["domain"]
            if dom not in ALLOWED_DOMAINS:
                continue
            kv = c.get("kv") or []
            if not kv:
                continue

            gold = db.query(dom, kv)
            gold_names = {
                (r.get("name") or "").strip().lower()
                for r in gold
                if isinstance(r.get("name"), str)
            }
            gold_names.discard("")
            if not gold_names:
                continue

            pred_names = pred_index.get((did, turn, dom), [])

            for K in K_LIST:
                preds_set = set(pred_names[:K])
                tp = len(preds_set & gold_names)
                p  = len(preds_set)
                g  = len(gold_names)

                overall[K]["tp"] += tp
                overall[K]["p"]  += p
                overall[K]["g"]  += g
                overall[K]["samples"] += 1

                bydom[dom][K]["tp"] += tp
                bydom[dom][K]["p"]  += p
                bydom[dom][K]["g"]  += g
                bydom[dom][K]["samples"] += 1

                rr = reciprocal_rank_at_k(pred_names, gold_names, K)
                overall_rr_sum[K] += rr
                bydom_rr_sum[dom][K] += rr

    print("===== Overall =====")
    for K in K_LIST:
        o = overall[K]
        P, R, F1 = prf(o["tp"], o["p"], o["g"])
        mrr = overall_rr_sum[K] / o["samples"] if o["samples"] else 0.0
        print(
            f"@{K}  Samples={o['samples']:>6}  TP/Pred/Gold={o['tp']}/{o['p']}/{o['g']}  "
            f"P={P:.4f}  R={R:.4f}  F1={F1:.4f}  MRR={mrr:.4f}"
        )

    print("\n===== By Domain =====")
    for dom in sorted(ALLOWED_DOMAINS):
        for K in K_LIST:
            s = bydom[dom][K]
            P, R, F1 = prf(s["tp"], s["p"], s["g"])
            mrr = (
                bydom_rr_sum[dom][K] / s["samples"]
                if s["samples"] else 0.0
            )
            print(
                f"[{dom:10s}] @{K}  Samples={s['samples']:>6}  "
                f"TP/Pred/Gold={s['tp']}/{s['p']}/{s['g']}  "
                f"P={P:.4f}  R={R:.4f}  F1={F1:.4f}  MRR={mrr:.4f}"
            )


if __name__ == "__main__":
    evaluate()
