"""将 Kaputt query + reference parquet 数据合并输出为 JSON。

每条 query 保留全部字段，对应的 reference 按 item_identifier 关联，
reference_image / reference_crop / reference_mask 以列表形式存储。
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
from pathlib import Path

from json_process import write_json


def build_dataset_json(data_root: str | Path, split: str = "sample") -> list[dict]:
    data_root = Path(data_root)
    query_path = data_root / f"query-{split}.parquet"
    ref_path = data_root / f"reference-{split}.parquet"

    if not query_path.exists():
        raise FileNotFoundError(f"Query parquet not found: {query_path}")

    query_df = pd.read_parquet(query_path)
    # random sample 500 rows
    # query_df = query_df.sample(n=500, random_state=42)
    ref_df = pd.read_parquet(ref_path) if ref_path.exists() else pd.DataFrame()

    # Pre-group references by item_identifier
    ref_groups = {}
    if len(ref_df) > 0:
        for item_id, group in ref_df.groupby("item_identifier"):
            ref_groups[item_id] = {
                "reference_image": group["reference_image"].tolist(),
                "reference_crop": group["reference_crop"].tolist(),
                "reference_mask": group["reference_mask"].tolist(),
            }

    results = []
    idx = 0
    for _, row in query_df.iterrows():
        item = row.to_dict()
        # Convert numpy types to native Python types for JSON serialization
        for k, v in item.items():
            if hasattr(v, "item"):
                item[k] = v.item()

        # Attach reference lists
        refs = ref_groups.get(row.item_identifier, {})
        item["reference_image"] = refs.get("reference_image", [])
        item["reference_crop"] = refs.get("reference_crop", [])
        item["reference_mask"] = refs.get("reference_mask", [])
        item["idx"] = idx
        idx += 1
        results.append(item)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Kaputt dataset JSON from parquet files")
    parser.add_argument("--data_root", type=str, default="./data/kaputt/sample-data")
    parser.add_argument("--split", type=str, default="sample")
    parser.add_argument("--output", type=str, default="./data/kaputt/dataset.json")
    args = parser.parse_args()

    results = build_dataset_json(args.data_root, args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output_path), results)

    print(f"共 {len(results)} 条数据，已保存到 {output_path}")