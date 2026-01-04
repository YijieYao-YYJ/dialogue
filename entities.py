import json
from pathlib import Path

# -------------------- Config --------------------
DB_DIR = Path("db")
OUT = Path("results/entities/entities.json")

# 只保留三类
ALLOWED_DOMAINS = {"restaurant", "hotel", "attraction"}

# 每个域保留在 text 中的关键字段
DOMAIN_FIELDS = {
    "restaurant": ["food", "area", "pricerange"],
    "hotel":      ["area", "pricerange", "stars", "parking", "internet", "wifi", "type"],
    # "attraction": ["type", "area", "pricerange", "entrance_fee", "openhours"],
    "attraction": ["type", "area", "pricerange", "entrance_fee"],
}
# ------------------------------------------------


def normalize_val(v):

    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"yes", "y", "true"}:
        return "yes"
    if s in {"no", "n", "false"}:
        return "no"
    if s in {"?", "n/a", "na", "none", "unknown", ""}:
        return None
    # return s.replace("centre", "center")
    return s


def normalize_key(k: str) -> str:
    return (
        k.strip()
         .lower()
         .replace(" ", "_")
         .replace("/", "_")
         .replace("-", "_")
         .replace("__", "_")
    )


def flatten_attrs(domain: str, ent: dict) -> dict:

    attrs = {}
    for k, v in ent.items():
        nk = normalize_key(k)
        if nk == "location":
            continue
        if nk == "price" and isinstance(v, dict):

            for rk, rv in v.items():
                attrs[f"price_{normalize_key(rk)}"] = normalize_val(rv)
        else:
            attrs[nk] = normalize_val(v)

    # 如果 internet=yes，则补 wifi=yes
    if attrs.get("internet") == "yes":
        attrs["wifi"] = "yes"

    return attrs


def serialize_text(domain: str, attrs: dict) -> str:

    fields = DOMAIN_FIELDS.get(domain, [])
    parts = []

    for k in fields:
        v = attrs.get(k)
        if v not in (None, ""):
            parts.append(f"{k}: {v}")


    for k in ("price_single", "price_double", "price_family"):
        v = attrs.get(k)
        # if v not in (None, ""):
        #     parts.append(f"{k}: {v}")

    return "; ".join(parts)


def main():
    count = 0
    seen_ids = set()

    with OUT.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for p in sorted(DB_DIR.iterdir()):
            if not p.is_file() or not p.name.endswith("_db.json"):
                continue

            domain = p.stem.replace("_db", "")
            if ALLOWED_DOMAINS is not None and domain not in ALLOWED_DOMAINS:
                continue

            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue

            for i, ent in enumerate(data):
                attrs = flatten_attrs(domain, ent)
                text = serialize_text(domain, attrs)

                orig_id = attrs.get("id") or str(i)
                doc_id = f"{domain}#{orig_id}"
                if doc_id in seen_ids:
                    doc_id = f"{domain}#{orig_id}_{i}"
                seen_ids.add(doc_id)

                obj = {
                    "doc_id": doc_id,
                    "domain": domain,
                    "text": text,
                    "attrs": attrs,
                }
                if not first:
                    f.write(",\n")
                f.write(json.dumps(obj, ensure_ascii=False))
                first = False
                count += 1

        f.write("\n]\n")

    print(f"写出实体语料 -> {OUT}，共 {count} 条。")


if __name__ == "__main__":
    main()
