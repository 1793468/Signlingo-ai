"""
Robustness checks the training pipeline doesn't cover: what happens when
the model gets audio that doesn't look like the clean Kaggle dataset it
was trained on.

Cases covered:
  1. Pure silence            — should NOT confidently hallucinate text
  2. White noise              — should NOT confidently hallucinate text
  3. Very short clip (<0.5s)  — should not crash
  4. Real test audio + white noise injected at a few SNR levels
     — measures how much WER degrades as audio gets noisier

This doesn't replace src/benchmark.py (clean-set WER/CER) — it's a
smoke test for failure modes that a clean-audio benchmark can't surface.

Usage:
    python tests/test_robustness.py \
        --model_dir checkpoints/merged_phase2 \
        --test_manifest data/manifests/test.jsonl \
        --num_samples 10
"""
import argparse
import json
import random
from pathlib import Path

import evaluate
import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import normalize_arabic  # noqa: E402

wer_metric = evaluate.load("wer")


def load_model(model_dir, device):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.max_length = None
    return processor, model


def transcribe(processor, model, audio, sample_rate, device):
    inputs = processor.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="arabic",
            task="transcribe",
            max_new_tokens=225,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_beams=5,
        )
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def add_noise(audio, snr_db):
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    return (audio + noise).astype(np.float32)


def test_silence(processor, model, device, sample_rate):
    print("\n=== Case 1: pure silence (3s) ===")
    audio = np.zeros(sample_rate * 3, dtype=np.float32)
    pred = transcribe(processor, model, audio, sample_rate, device)
    print(f"Prediction: {pred!r}")
    if pred.strip():
        print("⚠️  FLAG: model produced non-empty text for silence — potential hallucination.")
    else:
        print("OK: empty/near-empty output for silence.")
    return {"case": "silence", "prediction": pred, "flagged": bool(pred.strip())}


def test_white_noise(processor, model, device, sample_rate):
    print("\n=== Case 2: pure white noise (3s) ===")
    audio = np.random.normal(0, 0.05, sample_rate * 3).astype(np.float32)
    pred = transcribe(processor, model, audio, sample_rate, device)
    print(f"Prediction: {pred!r}")
    if pred.strip():
        print("⚠️  FLAG: model produced non-empty text for white noise — potential hallucination.")
    else:
        print("OK: empty/near-empty output for white noise.")
    return {"case": "white_noise", "prediction": pred, "flagged": bool(pred.strip())}


def test_short_clip(processor, model, device, sample_rate):
    print("\n=== Case 3: very short clip (0.2s) ===")
    audio = np.random.normal(0, 0.02, int(sample_rate * 0.2)).astype(np.float32)
    try:
        pred = transcribe(processor, model, audio, sample_rate, device)
        print(f"Prediction: {pred!r}")
        print("OK: no crash on short input.")
        return {"case": "short_clip", "prediction": pred, "crashed": False}
    except Exception as e:
        print(f"⚠️  FLAG: crashed on short input: {e}")
        return {"case": "short_clip", "error": str(e), "crashed": True}


def test_noisy_real_audio(processor, model, device, sample_rate, test_manifest, num_samples):
    print(f"\n=== Case 4: real test audio at varying SNR (n={num_samples}) ===")
    records = [json.loads(l) for l in open(test_manifest, encoding="utf-8")]
    random.seed(42)
    sample = random.sample(records, min(num_samples, len(records)))

    results = {}
    for snr in [None, 20, 10, 5]:
        label = "clean" if snr is None else f"{snr}dB SNR"
        preds, refs = [], []
        for rec in sample:
            audio, _ = librosa.load(rec["audio_filepath"], sr=sample_rate)
            if snr is not None:
                audio = add_noise(audio, snr)
            pred = transcribe(processor, model, audio, sample_rate, device)
            preds.append(normalize_arabic(pred))
            refs.append(normalize_arabic(rec["text"]))
        pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
        if pairs:
            p, r = zip(*pairs)
            wer = wer_metric.compute(predictions=list(p), references=list(r))
        else:
            wer = None
        print(f"  {label:>10}: WER = {wer}")
        results[label] = wer
    return {"case": "noisy_real_audio", "wer_by_condition": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_manifest", default="data/manifests/test.jsonl")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out", default="results/robustness_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(args.model_dir, device)

    report = []
    report.append(test_silence(processor, model, device, args.sample_rate))
    report.append(test_white_noise(processor, model, device, args.sample_rate))
    report.append(test_short_clip(processor, model, device, args.sample_rate))
    report.append(
        test_noisy_real_audio(
            processor, model, device, args.sample_rate, args.test_manifest, args.num_samples
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
