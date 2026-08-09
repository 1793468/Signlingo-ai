"""
Scan the raw KARSL-502 dataset folders, extract sign IDs from folder names,
and produce stratified train/val/test metadata CSVs consumed by
`landmark_extraction.py`.

Owner: Mariam Ashraf Tobar

Usage:
    python -m src.preprocessing.prepare_metadata --config configs/config.yaml
"""
import argparse
import os
import re

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Sign-ID ranges kept for the current training subset (numbers, days, some letters/words —
# adjust to your own curated list). Full dataset covers 1..502.
SELECTED_RANGES = [
    (1, 6),
    (160, 217),
    (223, 243),
    (273, 294),
    (298, 299),
    (485, 495),
]


def is_selected(sign_id: int) -> bool:
    return any(start <= sign_id <= end for start, end in SELECTED_RANGES)


def collect_sequences(dataset_root: str, max_signs: int):
    """Walk the dataset root; every folder containing .jpg frames is one sequence."""
    rows = []
    for root, _dirs, files in os.walk(dataset_root):
        jpg_files = [f for f in files if f.endswith(".jpg")]
        if not jpg_files:
            continue

        path_str = " ".join(root.split(os.sep))
        match = re.search(r"(\d{4})", path_str)
        if not match:
            continue

        sign_id = int(match.group(1))
        if 1 <= sign_id < max_signs and is_selected(sign_id):
            rows.append({"sequence_path": root, "sign_id": sign_id})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = collect_sequences(cfg["data"]["raw_dataset_root"], cfg["data"]["max_signs"])
    print(f"Collected {len(df)} sequences across {df['sign_id'].nunique()} signs")

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["sign_id"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["sign_id"], random_state=42)

    out_paths = cfg["data"]["metadata_csv"]
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        os.makedirs(os.path.dirname(out_paths[split_name]), exist_ok=True)
        split_df.to_csv(out_paths[split_name], index=False)
        print(f"{split_name}: {len(split_df)} sequences -> {out_paths[split_name]}")


if __name__ == "__main__":
    main()
