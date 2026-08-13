"""
Owner: Mariam.

Merge a trained LoRA adapter into the base Whisper weights, producing a
single standalone model directory ready to export/deploy — no PEFT
dependency needed at inference time.

Usage:
    python src/merge_and_export.py \
        --base_model openai/whisper-small \
        --adapter_dir speech-to-text/checkpoints/lora_adapter \
        --out_dir speech-to-text/checkpoints/merged
"""
import argparse

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="openai/whisper-small")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    print("Merging adapter into base weights ...")
    model = model.merge_and_unload()

    processor = WhisperProcessor.from_pretrained(args.adapter_dir)

    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Merged model saved to {args.out_dir}")
    print("This directory is a standard Whisper model — no PEFT needed to load it.")


if __name__ == "__main__":
    main()
