#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


INPUT_PATH_DEFAULT = Path("MultiWOZ_2.1/train/dialogues_001.json")
OUTPUT_PATH_DEFAULT = Path("search.json")

def normalize_id(dialogue_id: Any) -> Any:
    """去除 .json 尾缀的归一化；非字符串则原样返回。"""
    if not isinstance(dialogue_id, str):
        return dialogue_id
    return dialogue_id[:-5] if dialogue_id.endswith(".json") else dialogue_id

def load_json(path: Path):
    # Python 3.7+ 会保留 JSON 中对象键的原始顺序
    return json.loads(path.read_text(encoding="utf-8"))

def find_dialogue_in_dict(data: Dict[str, Any], target_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:

    # 精确匹配 key
    if target_id in data:
        return target_id, data[target_id]

    # 归一化匹配 key
    norm_target = normalize_id(target_id)
    for k, v in data.items():
        if normalize_id(k) == norm_target:
            return k, v

    for k, v in data.items():
        if isinstance(v, dict):
            did = v.get("dialogue_id")
            if did == target_id or normalize_id(did) == norm_target:
                return k, v

    return None, None

def prompt_dialogue_id() -> str:
    try:
        did = input("请输入 dialogue_id（例如 SNG01856.json 或 SNG01856）：").strip()
    except EOFError:
        did = ""
    return did

def main():
    ap = argparse.ArgumentParser(
        description="【MultiWOZ 2.1 专用】从顶层为 dict 的 data.json 中导出指定对话到 JSON 文件。"
    )
    ap.add_argument("--id", "-i", help="dialogue_id（可带或不带 .json）")
    ap.add_argument("--input", "-I", default=str(INPUT_PATH_DEFAULT), help="输入 JSON 文件路径（默认 data.json）")
    ap.add_argument("--output", "-o", default=str(OUTPUT_PATH_DEFAULT), help="输出 JSON 文件路径（默认 search.json）")
    args = ap.parse_args()

    target_id = args.id or prompt_dialogue_id()
    if not target_id:
        print("❌ 未提供 dialogue_id，已退出。")
        sys.exit(1)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"❌ 输入文件不存在: {in_path}")
        sys.exit(1)

    try:
        data = load_json(in_path)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print("❌ 输入文件顶层应为 dict（MultiWOZ 2.1 结构）。")
        sys.exit(1)

    found_key, found_obj = find_dialogue_in_dict(data, target_id)
    if found_obj is None:
        payload = {"error": "dialogue_id not found", "dialogue_id": target_id}
        out_text = json.dumps(payload, ensure_ascii=False, indent=2, separators=(", ", ": "))
        out_path.write_text(out_text, encoding="utf-8")
        print(f"⚠️ 未找到 dialogue_id={target_id}，已写入 {out_path}（包含错误信息）。")
        sys.exit(0)

    # 输出与 2.1 原始风格一致：{ "<key>": {...} }
    wrapper = {found_key: found_obj}
    out_text = json.dumps(wrapper, ensure_ascii=False, indent=2, separators=(", ", ": "))
    out_path.write_text(out_text, encoding="utf-8")
    print(f"✅ 已写入 {out_path}（导出 {{'{found_key}': ...}}）")

if __name__ == "__main__":
    main()




