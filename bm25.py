import json
import math
import re
from pathlib import Path
from collections import Counter


QUERIES_PATH  = Path("queries_dev.json")
ENTITIES_PATH = Path("entities.json")
OUT_PATH      = Path("bm25_results.json")

ALLOWED_DOMAINS = {"restaurant", "hotel", "attraction"}
TOPK = 5
K1   = 1.2
B    = 0.75


_token_re = re.compile(r"[a-z0-9]+")

def tokenize(text: str):
    if not text:
        return []
    text = text.lower()
    # 统一
    text = text.replace("centre", "center")
    text = text.replace("moderately", "moderate")
    text = text.replace("night club", "nightclub")
    text = text.replace("swimming pool", "swimmingpool")
    text = text.replace("concert hall", "concerthall")
    text = text.replace("museums", "museum")
    text = text.replace("colleges", "college")
    text = text.replace("hotels", "hotel")
    text = text.replace("guest house", "guesthouse")
    text = text.replace("guest houses", "guesthouse")
    text = text.replace("mid price", "moderate")
    text = text.replace("downtown", "center")
    text = text.replace("inexpensive", "cheap")
    text = text.replace("cheaply", "cheap")

    return _token_re.findall(text)

# -------------------- Load entities --------------------
def load_entities(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data if (e.get("domain") in ALLOWED_DOMAINS)]


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
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for w, n in self.df.items()}
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

    def query(self, q_text, allowed_domains=None, topk=5):
        q_tokens = tokenize(q_text)
        scores = []
        for i, doc in enumerate(self.docs):
            if allowed_domains is not None and doc.get("domain") not in allowed_domains:
                continue
            s = self.score(q_tokens, i)
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:topk]


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


def main():
    entities = load_entities(ENTITIES_PATH)
    if not entities:
        print("No entities loaded. Check entities.json and ALLOWED_DOMAINS.")
        OUT_PATH.write_text("[]", encoding="utf-8")
        return

    bm25 = BM25Index(entities, get_text=lambda d: d.get("text", ""))
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    results = []
    # for q in queries:
    #     domains = q.get("domains") or []
    #     domains = [d for d in domains if d in ALLOWED_DOMAINS]
    #     if not domains:
    #         continue  # 此 USER 轮不检索
    #
    #     q_text = q.get("query_text", "")
    #     per_domain = []
    #     for dom in _unique_keep_order(domains):
    #         hits = bm25.query(q_text, allowed_domains={dom}, topk=TOPK)
    #         per_domain.append({
    #             "domain": dom,
    #             "topk": pack_top_items(entities, hits)
    #         })
    #
    #     results.append({
    #         "dialogue_id": q.get("dialogue_id"),
    #         "turn": q.get("turn"),
    #         "domains": _unique_keep_order(domains),
    #         "results": per_domain
    #     })
    for q in queries:
        # 仅当 constraints 非空才检索
        cons_list = q.get("constraints") or []
        if not cons_list:
            continue  # 此 USER 轮不检索

        # 域来源改为 constraints 中的 domain
        domains = [c.get("domain") for c in cons_list if isinstance(c, dict)]
        domains = [d for d in domains if d in ALLOWED_DOMAINS]
        domains = _unique_keep_order(domains)
        if not domains:
            continue  # 无有效域则不检索

        q_text = q.get("query_text", "")
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
    print(f"✅ 完成检索，写出 {len(results)} 条结果 -> {OUT_PATH}")


if __name__ == "__main__":
    main()


