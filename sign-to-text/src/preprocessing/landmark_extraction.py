"""
Extract hand-landmark sequences from KARSL-502 video frames using MediaPipe
Holistic, batched with resume support (safe to interrupt/restart on Kaggle).

Owner: Mariam Ashraf Tobar

Dataset source: https://www.kaggle.com/datasets/yousefdotpy/karsl-502
Download locally with:
    kaggle datasets download -d yousefdotpy/karsl-502 -p data/raw --unzip

Each sample folder under the raw dataset contains a sequence of .jpg frames
for one repetition of one sign. This script walks the metadata split
(train/val/test), runs MediaPipe on every frame of every sequence, and saves
batched .npz files of shape:
    X: object array of (T, 126) float16 landmark sequences
    y: int16 array of sign labels

Usage:
    python -m src.preprocessing.landmark_extraction --config configs/config.yaml --split train
"""
import argparse
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

BATCH_SIZE = 500
NUM_WORKERS = 2
BATCHES_PER_RUN = 5  # Kaggle sessions time out — process a few batches, then re-run to resume

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_landmarks_on_image(image, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def process_single_row(row_data):
    """Run MediaPipe Holistic over every frame in one sequence folder."""
    path, sign_id, seq_index = row_data

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        image_files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")
        )

        landmarks_seq = []
        example_frame, example_results = None, None

        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if example_frame is None:
                example_frame, example_results = frame.copy(), results

            lh = (
                np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
                if results.left_hand_landmarks
                else np.zeros(63)
            )
            rh = (
                np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
                if results.right_hand_landmarks
                else np.zeros(63)
            )
            landmarks_seq.append(np.concatenate([lh, rh]))

        return np.array(landmarks_seq, dtype=np.float16), sign_id, seq_index, example_frame, example_results


def get_existing_batches(output_dir: str):
    return sorted(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(output_dir)
        if f.startswith("batch_") and f.endswith(".npz")
    )


def run_extraction_with_resume(df: pd.DataFrame, split_name: str, base_save_dir: str, batch_size: int = BATCH_SIZE):
    output_dir = os.path.join(base_save_dir, split_name)
    landmark_img_dir = os.path.join(base_save_dir, "landmark_examples")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(landmark_img_dir, exist_ok=True)

    data = list(zip(df["sequence_path"], df["sign_id"], range(len(df))))
    total_batches = (len(data) + batch_size - 1) // batch_size
    existing_batches = set(get_existing_batches(output_dir))

    print(f"\nSplit: {split_name}")
    print(f"Total sequences: {len(data)} | Total batches: {total_batches}")

    start_time = time.time()
    processed_this_run = 0

    for batch_id in range(total_batches):
        if batch_id in existing_batches:
            continue
        if processed_this_run >= BATCHES_PER_RUN:
            print("\nBatch limit reached for this run — re-run the script to resume.")
            sys.exit(0)

        batch_start = time.time()
        start_idx = batch_id * batch_size
        batch_subset = data[start_idx : start_idx + batch_size]

        print(f"\nProcessing batch {batch_id}")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(executor.map(process_single_row, batch_subset), total=len(batch_subset)))

        # Save one annotated example frame per batch for a quick visual sanity check
        ex_idx = np.random.randint(len(results))
        _, ex_sign, _, ex_frame, ex_results = results[ex_idx]
        if ex_frame is not None and ex_results is not None:
            draw_landmarks_on_image(ex_frame, ex_results)
            cv2.imwrite(os.path.join(landmark_img_dir, f"batch_{batch_id}_sign_{ex_sign}.jpg"), ex_frame)

        X_batch = [r[0] for r in results]
        y_batch = [r[1] for r in results]
        np.savez_compressed(
            os.path.join(output_dir, f"batch_{batch_id}.npz"),
            X=np.array(X_batch, dtype=object),
            y=np.array(y_batch, dtype=np.int16),
        )

        print(f"Saved batch_{batch_id}.npz | batch time: {format_time(time.time() - batch_start)}")
        processed_this_run += 1
        del results, X_batch, y_batch
        gc.collect()

    print(f"\nFinished '{split_name}' — total time: {format_time(time.time() - start_time)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_save_dir = cfg["data"]["processed_dir"]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split_name in splits:
        # Each CSV has columns: sequence_path, sign_id (produced by the train/val/test split step)
        split_df = pd.read_csv(cfg["data"]["metadata_csv"][split_name])
        run_extraction_with_resume(split_df, split_name, base_save_dir)


if __name__ == "__main__":
    main()
