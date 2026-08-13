"""
Run the full held-out test-set evaluation and record actual WER/CER —
the number that's been missing so far (results/README.md only has 5
qualitative examples, not a real metric).

Usage:
    python src/benchmark.py \
        --model_dir checkpoints/merged_phase2 \
        --test_manifest data/manifests/test.jsonl \
        --out results/eval_metrics.json
"""
import argparse
import json
import time
from pathlib import Path

import evaluate
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset import ManifestSpeechDataset, WhisperDataCollator, normalize_arabic

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Merged model dir (plain HF Whisper)")
    parser.add_argument("--test_manifest", default="data/manifests/test.jsonl")
    parser.add_argument("--out", default="results/eval_metrics.json")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Cap test set for a quick run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir} on {device} ...")
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.max_length = None  # avoid the max_new_tokens/max_length warning

    dataset = ManifestSpeechDataset(args.test_manifest, processor, sample_rate=args.sample_rate)
    if args.limit:
        dataset.records = dataset.records[: args.limit]
    print(f"Evaluating on {len(dataset)} test samples")

    collator = WhisperDataCollator(processor=processor)

    predictions, references, durations = [], [], []
    start = time.time()

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Running inference"):
        batch_items = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        batch = collator(batch_items)
        input_features = batch["input_features"].to(device)

        t0 = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="arabic",
                task="transcribe",
                max_new_tokens=225,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                num_beams=args.num_beams,
            )
        durations.append((time.time() - t0) / len(batch_items))

        preds = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(normalize_arabic(p) for p in preds)
        references.extend(normalize_arabic(r) for r in refs)

    total_time = time.time() - start

    # Drop empty references — they make WER undefined for that pair.
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    empty_refs = len(predictions) - len(pairs)
    preds_clean, refs_clean = zip(*pairs) if pairs else ([], [])

    wer = wer_metric.compute(predictions=list(preds_clean), references=list(refs_clean))
    cer = cer_metric.compute(predictions=list(preds_clean), references=list(refs_clean))

    # Also flag empty predictions — the model producing nothing for real audio
    # is a distinct failure mode from getting words wrong.
    empty_preds = sum(1 for p in preds_clean if not p.strip())

    results = {
        "model_dir": args.model_dir,
        "test_manifest": args.test_manifest,
        "num_samples": len(dataset),
        "num_evaluated": len(pairs),
        "num_empty_references_skipped": empty_refs,
        "num_empty_predictions": empty_preds,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "avg_inference_sec_per_sample": round(sum(durations) / len(durations), 3) if durations else None,
        "total_eval_time_sec": round(total_time, 1),
        "num_beams": args.num_beams,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
