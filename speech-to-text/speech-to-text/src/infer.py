"""
Run inference with a base Whisper model + a trained LoRA adapter.

Usage:
    python src/infer.py --audio path/to/clip.wav \
        --base_model openai/whisper-small \
        --adapter_dir speech-to-text/checkpoints/lora_adapter
"""
import argparse

import librosa
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe(audio_path, base_model, adapter_dir, sample_rate=16000):
    processor = WhisperProcessor.from_pretrained(adapter_dir)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Explicit language/task on the model's generation_config — without
    # this Whisper can default to translate-to-English on some inputs
    # instead of transcribing Arabic.
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    # generation_config ships with max_length=448 by default. Passing
    # max_new_tokens at generate() time overrides it anyway, but leaving
    # both set fires a "both max_new_tokens and max_length seem to have
    # been set" warning on every call — clear it so logs stay clean.
    model.generation_config.max_length = None

    audio, _ = librosa.load(audio_path, sr=sample_rate)
    inputs = processor.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="arabic",          # reminded again at generate() time
            task="transcribe",          # transcribe, do NOT translate
            max_new_tokens=225,
            repetition_penalty=1.1,     # confirmed: 2.0 broke valid Arabic repetition
            no_repeat_ngram_size=3,
            num_beams=5,                # beam search — noticeably fewer hallucinated words than greedy
        )

    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--base_model", default="openai/whisper-small")
    parser.add_argument("--adapter_dir", default="speech-to-text/checkpoints/lora_adapter")
    args = parser.parse_args()

    text = transcribe(args.audio, args.base_model, args.adapter_dir)
    print(text)


if __name__ == "__main__":
    main()
