# speech-to-text

Egyptian Arabic speech → text, via Whisper + LoRA. Self-contained sibling
pipeline to [`sign-to-text/`](../sign-to-text), following the same
data → preprocess → train → infer structure.

**Source of truth:** [`notebooks/egyptian-stt-baseline-30k-clean.ipynb`](notebooks/egyptian-stt-baseline-30k-clean.ipynb)
— the actual Kaggle training run (single T4 GPU). The scripts under
`data/` and `src/` mirror that notebook's logic in importable/runnable
form; if the two ever disagree, the notebook is what was really run.

## Dataset

[400K Egyptian Arabic Lines](https://www.kaggle.com/datasets/fadisarwat/egyptian-arabic-lines)
(Kaggle). Confirmed layout (from the notebook):

```
egyptian-arabic-lines/
├── index.csv     # columns include: audio_file, text, (gender)
└── data/         # audio referenced by index.csv's audio_file column
```

Not redistributed in this repo — download it yourself via the Kaggle API
(see below), and respect the dataset's license on Kaggle.

Training used a **stratified 30k subsample** of the 400k lines (by
gender, where available), not the full dataset — see `MAX_SAMPLES` in
the notebook / `dataset.max_samples` in `configs/config.yaml`.

## Team

| Owner | Responsibility |
| --- | --- |
| **Mariam** | Data pipeline, LoRA/model design, `src/train_phase2.py` (encoder unfreeze), evaluation |
| **Youstina** | `src/train_phase1.py`, code review, `merge_and_export.py` / `export_onnx.py`, backend integration, test cases |

Phase 1 → Phase 2 is a real handoff: Youstina's `train_phase1.py` produces
`checkpoints/lora_adapter/`, which Mariam's `train_phase2.py` loads and
continues from. Agree on where that checkpoint lands (shared Kaggle
output, Drive, etc.) before kicking off a full run.

## Layout

```
speech-to-text/
├── notebooks/
│   └── egyptian-stt-baseline-30k-clean.ipynb   # actual training run — source of truth
├── configs/
│   └── config.yaml         # dataset paths, model, LoRA, two-phase training hyperparams
├── data/
│   ├── download_dataset.py # pulls the Kaggle dataset via kagglehub
│   ├── preprocess.py       # builds train/val/test manifests (jsonl), applies normalize_arabic
│   ├── raw/                # (gitignored) downloaded dataset
│   └── manifests/          # (gitignored) generated manifests
├── src/
│   ├── dataset.py          # PyTorch Dataset + collator for Whisper (matches notebook's collator)
│   ├── train_phase1.py     # Phase 1: encoder frozen, LoRA only          — owner: Youstina
│   ├── train_phase2.py     # Phase 2: encoder unfrozen, continued FT     — owner: Mariam
│   ├── merge_and_export.py # folds a standalone LoRA adapter into base weights
│   ├── export_onnx.py      # ONNX export for backend serving
│   └── infer.py            # transcribe a single audio file
├── checkpoints/            # (gitignored) trained adapters / merged models
└── requirements.txt
```

## Setup

```bash
cd speech-to-text
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Configure Kaggle credentials (needed for the download step):

```bash
# https://www.kaggle.com/settings -> Create New Token, downloads kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Pipeline

1. **Download the dataset**

   ```bash
   python data/download_dataset.py --out data/raw
   ```

2. **Build manifests** — the index file and audio subfolder are now
   confirmed (`index.csv` / `data/`), so this runs without guessing:

   ```bash
   python data/preprocess.py --raw_dir data/raw --out_dir data/manifests
   ```

   Text is normalized with `normalize_arabic()` at this stage — the same
   function the notebook applies to both training labels and eval
   predictions (diacritics stripped, alef/waw/ya variants unified, tah
   marbuta normalized, non-Arabic characters dropped). Keeping this in
   one place (rather than reimplementing it per-script) is what the
   notebook's own changelog calls out as a critical fix — a mismatch
   here silently inflates WER.

   For a quick smoke test first: add `--limit 2000`.

3. **Fine-tune with LoRA** — two phases, matching the notebook, split
   across two scripts so the Phase 1 → Phase 2 handoff is explicit:

   - **Phase 1** (Youstina) — encoder frozen, train decoder + LoRA
     adapters only (`q_proj`, `v_proj`, `k_proj`, `out_proj`; rank 32,
     alpha 64) at `lr=1e-5` for up to 4000 steps. Freezing the encoder
     first protects Whisper's pretrained acoustic features from being
     overwritten early.

     ```bash
     cd src
     python train_phase1.py --config ../configs/config.yaml
     ```

     Saves a LoRA adapter to `checkpoints/lora_adapter/` — hand this off.

   - **Phase 2** (Mariam) — loads Phase 1's adapter, merges it into the
     base weights, unfreezes the encoder, and continues fine-tuning the
     whole model at a lower `lr=5e-6` for up to 1000 more steps, so
     Egyptian Arabic phonetics get baked into the encoder without
     catastrophic forgetting.

     ```bash
     python train_phase2.py --config ../configs/config.yaml \
         --phase1_adapter ../checkpoints/lora_adapter
     ```

     Saves the final merged model to `checkpoints/merged_phase2/` — this
     is already a plain Whisper model, ready for `export_onnx.py`
     directly (no need to run `merge_and_export.py` again).

   Model defaults to `openai/whisper-small` — confirmed via the notebook
   that `whisper-medium` OOMs on a single T4 (15GB VRAM), so `small` is
   the safe default here, not just a speed tradeoff.

4. **Run inference** (against a standalone LoRA adapter, e.g. right
   after Phase 1, before Phase 2 has run)

   ```bash
   python src/infer.py --audio sample.wav \
       --adapter_dir checkpoints/lora_adapter
   ```

   Generation uses `repetition_penalty=1.1` and `no_repeat_ngram_size=3`
   — an earlier `repetition_penalty=2.0` was breaking valid Arabic
   repetition, per the notebook's changelog.

5. **Merge a standalone adapter** (only needed if you're merging outside
   the Phase 2 flow, e.g. testing Phase 1's adapter alone)

   ```bash
   python src/merge_and_export.py \
       --base_model openai/whisper-small \
       --adapter_dir checkpoints/lora_adapter \
       --out_dir checkpoints/merged
   ```

6. **Export to ONNX for backend integration** (Youstina)

   ```bash
   python src/export_onnx.py \
       --model_dir checkpoints/merged_phase2 \
       --out_dir checkpoints/onnx
   ```

   The notebook does this same export directly via
   `optimum.exporters.onnx.main_export(...)` — `export_onnx.py` wraps
   the equivalent `optimum-cli` call.

## Evaluation

WER and CER, computed with `normalize_arabic()` applied to both
predictions and references (the notebook's changelog flags this as a fix
— they weren't consistently normalized before, which inflated WER
artificially). Evaluate on the **full** held-out test split, not a
capped subset — an earlier version of the notebook capped eval at 500
samples, which the changelog also calls out as a bug.

## Integrating into the app

Same as before — this feeds the backend `/api/speech-to-text` endpoint,
which the Flutter app's `SpeechToTextApiService` (in `Graduation-project`)
already calls, alternative to the on-device `speech_to_text` package.

- **Backend does the transcribing** — serve `checkpoints/onnx` (or the
  merged HF model) from a small Python inference service, called from
  Laravel the way the mobile app's `SpeechToTextApiService` expects.
- **On-device** — would need a further conversion (`whisper.cpp` /
  `ctranslate2`); not attempted here.

## Status

- Phase 1 (`train_phase1.py`) and Phase 2 (`train_phase2.py`) are both
  implemented end-to-end, matching the notebook, and split across files
  along the Youstina/Mariam ownership boundary described above.
- Qualitative eval samples from the Phase 2 model are in
  [`results/README.md`](results/README.md) — gist is consistently
  correct, but with occasional hallucinated words and truncation on
  longer sentences. Not yet a WER/CER number — that needs the full
  test-set eval (notebook Step 13), which hasn't been run/recorded here.
- A Streamlit demo app (`app.py`, tunneled via localtunnel) exists in
  the notebook for manual testing on Kaggle — not part of this repo
  structure, kept in the notebook only.
