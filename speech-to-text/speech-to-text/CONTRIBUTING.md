# Contributing (speech-to-text)

Two people, one pipeline, staged handoff — same idea as [`sign-to-text`'s
CONTRIBUTING](../CONTRIBUTING.md), but **the roles are reversed here**:
on `sign-to-text`, Youstina builds the ONNX export and backend
integration. On `speech-to-text`, Mariam does.

## Roles

- **Youstina** builds: Phase 1 training (`train_phase1.py`, encoder
  frozen, LoRA only), ending in a LoRA adapter checkpoint.
- **Mariam** reviews Phase 1, then owns: Phase 2 training
  (`train_phase2.py`, encoder unfrozen), `merge_and_export.py`,
  `export_onnx.py`, and backend integration.
- **Youstina** reviews Mariam's stages in turn, and writes/runs the test
  cases (`tests/test_robustness.py`, plus the full-test benchmark in
  `src/benchmark.py`) against what Mariam built — she follows and
  validates this part rather than building it.

This is sequential, not two people racing on the same file — branches
map to stages, and every stage gets reviewed by the other person before
the next one builds on top of it.

## Branching

- `main` — always working, always the latest agreed-on state.
  Protected: no direct pushes.
- `feature/<short-description>` — one branch per stage, e.g.:
  - `feature/data-preprocessing`
  - `feature/phase1-lora-training`
  - `feature/phase2-encoder-finetune`
  - `feature/onnx-export`
  - `feature/backend-integration`
  - `feature/robustness-tests`

## Workflow

```bash
git checkout main
git pull
git checkout -b feature/phase2-encoder-finetune

# ...work, commit in small logical chunks...
git add src/train_phase2.py
git commit -m "Add Phase 2 encoder-unfreeze training script"

git push -u origin feature/phase2-encoder-finetune
# open a Pull Request on GitHub, request review from the other person
```

## Review handoff

- Youstina opens a PR for Phase 1 (data + preprocessing groundwork +
  `train_phase1.py`). Mariam reviews and merges.
- Once Phase 1 is merged and a LoRA adapter checkpoint exists, Mariam
  branches off `main` for `feature/phase2-encoder-finetune`, then
  `feature/onnx-export`, then `feature/backend-integration` — each its
  own PR, reviewed by Youstina before merging.
- Youstina's `feature/robustness-tests` branch runs against whatever
  Mariam's latest merged export is. If it surfaces a bug (e.g. the model
  hallucinating on silence, or an ONNX output shape mismatch with the
  backend), open an issue or a small follow-up PR rather than editing
  on top of an unmerged branch.

## Commit messages

Small, imperative, one topic per commit:

- `Add normalize_arabic to preprocessing manifest builder`
- `Fix max_length/max_new_tokens warning in generation config`
- `Add SNR-degradation robustness test`

Avoid `update`, `fix stuff`, `final final v2`.

## Pull Requests

- Every PR into `main` needs at least one review from the other AI
  teammate.
- Keep PRs scoped to one stage — easier to review, easier to revert if
  needed.
