# Sign-to-Text — Arabic Sign Language Recognition

Converts a sequence of Arabic Sign Language (ArSL) frames into text, using
hand-landmark sequences extracted with MediaPipe and classified with a
BiLSTM + Attention model.

## Overview

```
Video frames ──▶ MediaPipe Hand Landmarks ──▶ Normalize + Pad (30 frames) ──▶ BiLSTM + Attention ──▶ Predicted sign
                  (21 landmarks × 2 hands
                   = 126 features/frame)
```

- **Dataset:** [KARSL-502](https://www.kaggle.com/datasets/yousefdotpy/karsl-502) (Kaggle) — 502
  Arabic Sign Language signs. A curated subset of ~120 signs is used for the current training run
  (see `configs/config.yaml`).
- **Test accuracy (120-sign subset):** 97.72%

## Repo structure

```
sign-to-text/
├── src/
│   ├── preprocessing/
│   │   ├── landmark_extraction.py   # MediaPipe → .npz landmark sequences
│   │   └── normalize.py             # centering, scaling, padding to 30 frames
│   ├── datasets/
│   │   └── sign_dataset.py          # PyTorch Dataset over processed .npz files
│   ├── models/
│   │   └── lstm_attention.py        # BiLSTM + Attention model
│   ├── training/
│   │   └── train.py                 # training/eval loop
│   └── export/
│       └── export_onnx.py           # ONNX export + inference sanity check
├── configs/
│   └── config.yaml                  # paths, hyperparameters
├── notebooks/                       # exploratory notebooks only (not the source of truth)
├── tests/
│   └── test_preprocessing.py
├── models/                          # trained weights (gitignored, see note below)
├── data/                            # raw/processed data (gitignored, see note below)
├── requirements.txt
└── README.md
```

> **Why move code out of notebooks?** Notebooks (`notebooks/`) stay as scratch space for
> experiments and plots. The actual pipeline lives in `src/` as importable, testable modules —
> this is what reviewers/supervisors and your teammates will actually read, and it's what makes
> the work reviewable and testable rather than a single giant `.ipynb`.

## Setup

```bash
git clone https://github.com/1793468/signlingo-ai.git
cd signlingo-ai/sign-to-text
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

## Dataset

This project uses **[KARSL-502](https://www.kaggle.com/datasets/yousefdotpy/karsl-502)**,
published on Kaggle by [yousefdotpy](https://www.kaggle.com/yousefdotpy). It contains video
frames for 502 Arabic Sign Language signs performed by multiple signers.

To download it locally (requires a [Kaggle API token](https://www.kaggle.com/docs/api)):

```bash
pip install kaggle
# place your kaggle.json in ~/.kaggle/ first
kaggle datasets download -d yousefdotpy/karsl-502 -p data/raw --unzip
```

Then point `data.raw_dataset_root` in `configs/config.yaml` to `data/raw`.

> **Attribution:** if you publish results or the thesis report, cite the dataset owner/source
> (Kaggle: yousefdotpy/karsl-502) alongside the original KARSL paper if available.

## Usage

```bash
# 1. Build stratified train/val/test metadata CSVs from the raw KARSL-502 folders
python -m src.preprocessing.prepare_metadata --config configs/config.yaml

# 2. Extract MediaPipe landmarks into batched .npz sequences (safe to interrupt/resume)
python -m src.preprocessing.landmark_extraction --config configs/config.yaml

# 3. Train
python -m src.training.train --config configs/config.yaml

# 4. Export the trained model to ONNX
python -m src.export.export_onnx --checkpoint models/lstm_attention_best.pt
```

## Data & model weights

Raw datasets and trained weights are **not committed to git**. They're large,
change often, and don't belong in version control. Instead:
- `data/` and `models/` are in `.gitignore`.
- Datasets are pulled from Kaggle (`karsl-502`) — see `configs/config.yaml` for paths.
- Share trained checkpoints via a shared Google Drive / Kaggle Dataset / GitHub Release, and
  link them here once available.

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| BiLSTM + Attention | 97.72% | 120-sign subset, 30-frame sequences |

## Workflow / handoff

This pipeline is built and reviewed in stages rather than split into parallel independent parts:

1. **Mariam** builds preprocessing → dataset → BiLSTM+Attention model → training/eval, up through
   producing a trained checkpoint (`models/lstm_attention_best.pt`), each stage as its own PR.
2. **Youstina** reviews each PR, then owns:
   - Refining/hardening `src/export/export_onnx.py` and verifying the exported ONNX model matches
     PyTorch outputs (sanity-check script, not just "it exported without erroring").
   - Reviewing integration with the backend (how the model is served/called from the Laravel API
     described in the main SignLingo docs).
   - Writing test cases (`tests/`) — unit tests for preprocessing (already started), plus
     inference tests against the exported ONNX model.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the branch/PR conventions.

## Roadmap

- [ ] Expand from 120 → full 502 signs
- [ ] Quantize ONNX model for on-device (mobile) inference
- [ ] Two-hand overlap / occlusion robustness
- [ ] End-to-end test: raw frames in → predicted sign out, via the exported ONNX model
- [ ] Backend integration test (API request → model inference → response)
