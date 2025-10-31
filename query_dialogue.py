"""
MultiWOZ 2.1 触发（看下一轮 SYSTEM）

触发：
  - 下一 SYSTEM 存在 Domain-Recommend，或
  - 下一 SYSTEM 存在 Domain-Inform 且包含 Name 槽，或
  - 下一 SYSTEM 存在 Domain-Inform 且包含 Choice 槽

否决：
  - 当前 USER 的 Domain-Inform 含 Name 槽（值非空/非 not mentioned/dontcare/none/do n't care），则不触发该域

只输出 constraints：
  - 仅对被触发的 domains 生成；
  - 仅当该域抽到的约束（去掉 name 后）非空时才写入；
  - constraints 不包含 name 槽。
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any


INPUT_PATH  = Path("MultiWOZ_2.1/train/dialogues_001.json")
OUTPUT_PATH = Path("queries1.json")


ALLOWED_DOMAINS: Set[str] = {"restaurant", "hotel", "attraction"}

DOMAIN_SLOT = {
    "restaurant": ["food", "area", "pricerange"],
    "hotel":      ["area", "pricerange", "stars", "parking", "internet", "type"],
    "attraction": ["type", "area", "pricerange"],
}


def to_tagged_line(speaker: str, utterance: str) -> str:
    spk = speaker if speaker in ("USER", "SYSTEM") else "UNKNOWN"
    return f"[{spk}]{(utterance or '').strip()}"


def get_speaker_by_index(idx: int) -> str:
    return "USER" if idx % 2 == 0 else "SYSTEM"


_EMPTY_TOKENS = {"", "not mentioned", "dontcare", "do n't care", "none", "dont care", "don't care"}

def _is_empty_like(x: Any) -> bool:
    if x is None:
        return True
    if not isinstance(x, str):
        return False
    return x.strip().lower() in _EMPTY_TOKENS


def _act_slots(act_dict: Dict, domain: str, suffix: str) -> List[Tuple[str, str]]:
    """从 dialog_act 取 f"{Domain}-{suffix}"，返回 (slot, value) 列表。"""
    if not isinstance(act_dict, dict):
        return []
    want_key_lower = f"{domain}-{suffix}".lower()
    out: List[Tuple[str, str]] = []
    for k, pairs in act_dict.items():
        if not isinstance(k, str):
            continue
        if k.lower() == want_key_lower and isinstance(pairs, list):
            for p in pairs:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    slot = str(p[0]) if p[0] is not None else ""
                    val  = str(p[1]) if p[1] is not None else ""
                    out.append((slot, val))
    return out


def _has_slot_name(pairs: List[Tuple[str, str]]) -> bool:
    for slot, val in pairs:
        if slot.strip().lower() == "name" and not _is_empty_like(val):
            return True
    return False


def _has_choice_any(pairs: List[Tuple[str, str]]) -> bool:
    for slot, _ in pairs:
        if slot.strip().lower() == "choice":
            return True
    return False


def _domain_cap(dom: str) -> str:
    if not dom:
        return dom
    return dom[0].upper() + dom[1:].lower()


def _build_constraints_from_next_system(next_system_turn: Optional[Dict], domain: str) -> List[Tuple[str, str]]:
    """
    从下一轮 SYSTEM 的 metadata[domain]['semi'] 提取约束：
      - 仅取 DOMAIN_SLOT 中的槽位；
      - 过滤空值；
      - area: center -> centre；
      - 不包含 name。
    """
    if next_system_turn is None:
        return []
    meta = next_system_turn.get("metadata") or {}
    dom_meta = meta.get(domain) or {}
    semi = dom_meta.get("semi") or {}
    if not isinstance(semi, dict):
        return []

    keys = DOMAIN_SLOT.get(domain, [])
    kv: List[Tuple[str, str]] = []
    for k in keys:
        v = semi.get(k)
        if _is_empty_like(v):
            continue
        s = str(v).strip()
        if k == "area" and s.lower() == "center":
            s = "centre"
        kv.append((k, s))
    return kv


# ---------- 触发：看“当前 USER + 下一 SYSTEM” ----------
def collect_domains_user_turn_by_next(
    curr_user_turn: Dict,
    next_system_turn: Optional[Dict],
    allowed_domains: Set[str]
) -> List[str]:
    """
    逻辑：
      - 看下一 SYSTEM: 有 Domain-Recommend；或 Domain-Inform 含 Name；或 Domain-Inform 含 Choice
      - 否决：若当前 USER 的 Domain-Inform 含 Name，则不触发
    """
    if next_system_turn is None:
        return []

    user_act = curr_user_turn.get("dialog_act") or {}
    sys_act  = next_system_turn.get("dialog_act") or {}

    need: Set[str] = set()

    for dom in allowed_domains:
        dom_cap = _domain_cap(dom)

        # Recommend 触发
        sys_rec_pairs = _act_slots(sys_act, dom_cap, "Recommend")
        if sys_rec_pairs:
            need.add(dom)
            continue

        # Inform + (Name 或 Choice) 触发
        sys_inf_pairs = _act_slots(sys_act, dom_cap, "Inform")
        if sys_inf_pairs and (_has_slot_name(sys_inf_pairs) or _has_choice_any(sys_inf_pairs)):
            need.add(dom)

    # 否决：当前 USER 已给 Name
    for dom in list(need):
        dom_cap = _domain_cap(dom)
        user_inf_pairs = _act_slots(user_act, dom_cap, "Inform")
        if user_inf_pairs and _has_slot_name(user_inf_pairs):
            need.discard(dom)

    return sorted(need)



def parse_dialogues_with_next(obj: Dict[str, Dict], allowed_domains: Set[str]) -> List[Dict]:
    out: List[Dict] = []
    for did, d in obj.items():
        logs = d.get("log") or []
        buffer_lines: List[str] = []

        for idx, turn in enumerate(logs):
            speaker = get_speaker_by_index(idx)
            text = turn.get("text", "")
            buffer_lines.append(to_tagged_line(speaker, text))

            if speaker == "USER":
                next_sys = logs[idx + 1] if idx + 1 < len(logs) else None
                domains = collect_domains_user_turn_by_next(turn, next_sys, allowed_domains)

                # 仅为“被触发的域”抽取约束；且该域抽到的约束非空时才加入
                cons_list = []
                for dom in domains:
                    kv = _build_constraints_from_next_system(next_sys, dom)
                    if kv:  # 只有“有实体（非空约束）”时才写入
                        cons_list.append({"domain": dom, "kv": kv})

                item = {
                    "dialogue_id": did,
                    "turn": idx,
                    "domains": domains,                  # 触发到的域（可能非空）
                    "constraints": cons_list,            # 仅包含有约束的域
                    "query_text": "\n".join(buffer_lines)
                }
                out.append(item)

    return out


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    obj = {str(k): v for k, v in data.items()}

    out_items = parse_dialogues_with_next(obj, ALLOWED_DOMAINS)

    # 只保留“至少一个域写出了非空 constraints”的 USER 轮：
    # out_items = [x for x in out_items if x["constraints"]]

    OUTPUT_PATH.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已生成 {len(out_items)} 条查询 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
