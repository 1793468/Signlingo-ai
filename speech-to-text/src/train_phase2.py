"""
Phase 2 — unfreeze the encoder, continue fine-tuning the whole model at a
lower learning rate. Owner: Mariam.

Continues from Phase 1's LoRA adapter (train_phase1.py, owned by
Youstina). Merges that adapter into the base weights first, then
unfreezes the encoder and keeps training — bakes Egyptian Arabic
phonetics into the encoder without catastrophic forgetting. Matches
notebooks/egyptian-stt-baseline-30k-clean.ipynb Step 11.

Usage:
    python src/train_phase2.py --config configs/config.yaml \
        --phase1_adapter checkpoints/lora_adapter
"""
import argparse

import evaluate
import torch
import yaml
from peft import PeftModel
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from dataset import ManifestSpeechDataset, WhisperDataCollator, normalize_arabic
from train_phase1 import build_compute_metrics  # reuse the same WER/CER metric fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--phase1_adapter",
        default=None,
        help="Path to Phase 1's LoRA adapter. Defaults to <output_dir>/lora_adapter.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["base_model"]
    adapter_dir = args.phase1_adapter or f"{cfg['training']['output_dir']}/lora_adapter"

    print(f"Loading base model: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    print(f"Loading Phase 1 adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    print("Merging Phase 1 adapter into base weights...")
    model = model.merge_and_unload()

    processor = WhisperProcessor.from_pretrained(adapter_dir)
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    for param in model.model.encoder.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder unfrozen. Trainable params: {trainable / 1e6:.1f}M")

    manifest_dir = cfg["dataset"]["manifest_dir"]
    sr = cfg["dataset"]["sample_rate"]
    train_ds = ManifestSpeechDataset(f"{manifest_dir}/train.jsonl", processor, sample_rate=sr)
    val_ds = ManifestSpeechDataset(f"{manifest_dir}/val.jsonl", processor, sample_rate=sr)
    collator = WhisperDataCollator(processor=processor)

    p2 = cfg["training"]["phase2"]
    p1 = cfg["training"]["phase1"]  # reuse batch/eval sizing from phase1
    phase2_output = f"{cfg['training']['output_dir']}_phase2"

    training_args = Seq2SeqTrainingArguments(
        output_dir=phase2_output,
        per_device_train_batch_size=p1["per_device_train_batch_size"],
        per_device_eval_batch_size=p1["per_device_eval_batch_size"],
        gradient_accumulation_steps=p1["gradient_accumulation_steps"],
        learning_rate=p2["learning_rate"],
        warmup_steps=p2["warmup_steps"],
        max_steps=p2["max_steps"],
        gradient_checkpointing=True,
        bf16=p1["bf16"] and torch.cuda.is_available(),
        fp16=p1["fp16"],
        eval_strategy="steps",
        eval_steps=p1["eval_steps"],
        save_steps=p1["save_steps"],
        logging_steps=p1["logging_steps"],
        predict_with_generate=True,
        generation_max_length=cfg["training"]["generation_max_length"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["none"],
        remove_unused_columns=False,
        label_names=["labels"],
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    final_dir = f"{cfg['training']['output_dir']}/merged_phase2"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Phase 2 complete. Final merged model saved to {final_dir}")
    print("This is a plain Whisper model — feed it directly into export_onnx.py.")


if __name__ == "__main__":
    main()
