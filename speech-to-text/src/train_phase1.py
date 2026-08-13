"""
Phase 1 — encoder frozen, train decoder + LoRA adapters only.
Owner: Youstina.

Freezing the encoder first protects Whisper's pretrained acoustic
features from being overwritten early — matches
notebooks/egyptian-stt-baseline-30k-clean.ipynb Step 10.

Produces a LoRA adapter checkpoint that Phase 2 (train_phase2.py,
owned by Mariam) continues from — hand off checkpoints/lora_adapter/
when this finishes.

Usage:
    python src/train_phase1.py --config configs/config.yaml
"""
import argparse

import evaluate
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from dataset import ManifestSpeechDataset, WhisperDataCollator, normalize_arabic

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def build_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_arabic(s) for s in pred_str]
        label_str = [normalize_arabic(s) for s in label_str]

        pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        pred_str, label_str = zip(*pairs)

        wer = wer_metric.compute(predictions=list(pred_str), references=list(label_str))
        cer = cer_metric.compute(predictions=list(pred_str), references=list(label_str))
        return {"wer": round(wer, 4), "cer": round(cer, 4)}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["base_model"]
    processor = WhisperProcessor.from_pretrained(
        model_name, language=cfg["model"]["language"], task=cfg["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen for Phase 1")

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    manifest_dir = cfg["dataset"]["manifest_dir"]
    sr = cfg["dataset"]["sample_rate"]
    train_ds = ManifestSpeechDataset(f"{manifest_dir}/train.jsonl", processor, sample_rate=sr)
    val_ds = ManifestSpeechDataset(f"{manifest_dir}/val.jsonl", processor, sample_rate=sr)
    collator = WhisperDataCollator(processor=processor)

    p1 = cfg["training"]["phase1"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=p1["per_device_train_batch_size"],
        per_device_eval_batch_size=p1["per_device_eval_batch_size"],
        gradient_accumulation_steps=p1["gradient_accumulation_steps"],
        learning_rate=p1["learning_rate"],
        warmup_steps=p1["warmup_steps"],
        max_steps=p1["max_steps"],
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

    adapter_dir = f"{cfg['training']['output_dir']}/lora_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Phase 1 complete. LoRA adapter saved to {adapter_dir}")
    print("Hand this checkpoint to Phase 2 (train_phase2.py).")


if __name__ == "__main__":
    main()
