#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import OrderedDict

# ---------- 配置 ----------
DATA_PATH = Path("data.json")
VAL_LIST  = Path("valListFile.txt")
TEST_LIST = Path("testListFile.txt")

OUT_DIR   = Path(".")  # 输出根目录，会创建 train/ dev/ test/
NUM_FILES = {
    "train": 17,
    "dev":   2,
    "test":  2,
}
FILENAME_PREFIX = "dialogues_"
PAD_WIDTH = 3  # -> dialogues_001.json


def read_id_list(p: Path) -> set[str]:
    """读取 valListFile.txt / testListFile.txt"""
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def even_sequential_slices(n_items: int, n_files: int):
    """按顺序把 n_items 平均切为 n_files 份"""
    if n_items <= 0:
        return []
    n_files = max(1, min(n_files, n_items))
    base = n_items // n_files
    rem = n_items % n_files
    slices = []
    start = 0
    for i in range(n_files):
        size = base + (1 if i < rem else 0)
        end = start + size
        slices.append((start, end))
        start = end
    return slices


def write_shards(split_name: str, items: list[tuple[str, dict]], out_root: Path, n_files: int):
    """将 items 按顺序写出为若干个字典文件"""
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = even_sequential_slices(len(items), n_files)

    for i, (s, e) in enumerate(shards, start=1):
        shard_items = items[s:e]
        shard_dict = OrderedDict(shard_items)  # 保留顺序
        fname = f"{FILENAME_PREFIX}{str(i).zfill(PAD_WIDTH)}.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(shard_dict, f, indent=2, ensure_ascii=False)

    print(f"[{split_name}] total={len(items)} -> files={len(shards)}")
    for i, (s, e) in enumerate(shards, start=1):
        print(f"  {FILENAME_PREFIX}{str(i).zfill(PAD_WIDTH)}.json: {e - s} 对话")


def main():
    # 读取数据（OrderedDict 确保顺序）
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    val_ids  = read_id_list(VAL_LIST)
    test_ids = read_id_list(TEST_LIST)

    # 顺序遍历 data，按文件列表划分
    train_items, dev_items, test_items = [], [], []

    for k, v in data.items():
        if k in val_ids:
            dev_items.append((k, v))
        elif k in test_ids:
            test_items.append((k, v))
        else:
            train_items.append((k, v))

    # 写出三类文件
    write_shards("train", train_items, OUT_DIR, NUM_FILES["train"])
    write_shards("dev",   dev_items,   OUT_DIR, NUM_FILES["dev"])
    write_shards("test",  test_items,  OUT_DIR, NUM_FILES["test"])

    # 统计信息
    stats = {
        "train": len(train_items),
        "dev":   len(dev_items),
        "test":  len(test_items),
        "files_per_split": NUM_FILES,
    }
    (OUT_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n✅ Done. Stats saved to split_stats.json")


if __name__ == "__main__":
    main()



