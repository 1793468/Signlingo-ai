"""
Dataset + collator that turn manifest JSONL rows into Whisper model inputs.

The collator re-applies normalize_arabic() to labels at batch time — this
matches the training notebook exactly (notebooks/egyptian-stt-baseline-30k-clean.ipynb),
where the same normalization runs on both labels and predictions. If this
drifts out of sync between training and eval, WER numbers become meaningless.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import torch
from torch.utils.data import Dataset


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"[ىئ]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s\d]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ManifestSpeechDataset(Dataset):
    def __init__(self, manifest_path: str, processor, sample_rate: int = 16000):
        self.processor = processor
        self.sample_rate = sample_rate
        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx) -> Dict[str, Any]:
        rec = self.records[idx]
        audio, _ = librosa.load(rec["audio_filepath"], sr=self.sample_rate)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=self.sample_rate
        ).input_features[0]

        labels = self.processor.tokenizer(normalize_arabic(rec["text"])).input_ids

        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip the BOS token if the tokenizer prepended one; the model adds it back.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
