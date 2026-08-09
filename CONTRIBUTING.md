# Contributing (AI team)

Two people, one pipeline, staged handoff — here's how we work without stepping on each other.

## Roles

- **Mariam** builds: preprocessing → dataset → BiLSTM+Attention model → training/eval, ending in
  a trained checkpoint.
- **Youstina** reviews each stage, then owns: ONNX export refinement, backend-integration review,
  and test cases.

This is sequential, not two people racing on the same file — so branches map to *stages*, and
every stage gets reviewed by the other person before the next one builds on top of it.

## Branching

- `main` — always working, always the latest agreed-on state. Protected: no direct pushes.
- `feature/<short-description>` — one branch per stage, e.g.:
  - `feature/landmark-preprocessing`
  - `feature/lstm-attention-model`
  - `feature/training-eval`
  - `feature/onnx-export`
  - `feature/backend-integration-tests`
  - `feature/test-cases`

## Workflow

```bash
git checkout main
git pull
git checkout -b feature/lstm-attention-model

# ...work, commit in small logical chunks...
git add src/models/lstm_attention.py
git commit -m "Add BiLSTM+Attention model"

git push -u origin feature/lstm-attention-model
# open a Pull Request on GitHub, request review from the other person
```

## Review handoff

- Mariam opens a PR per stage (preprocessing, model, training). Youstina reviews and merges.
- Once training/eval is merged and a checkpoint exists, Youstina branches off `main` for
  `feature/onnx-export`, `feature/backend-integration-tests`, and `feature/test-cases` —
  each its own PR, reviewed by Mariam before merging.
- If Youstina's export/testing work surfaces a bug or a needed change upstream (e.g. in
  preprocessing or the model), open an issue or a small follow-up PR rather than editing on top
  of an unmerged branch.

## Commit messages

Small, imperative, one topic per commit:
- `Add hand-landmark normalization`
- `Fix padding for sequences shorter than 30 frames`
- `Add ONNX output sanity check against PyTorch model`

Avoid `update`, `fix stuff`, `final final v2`.

## Pull Requests

- Every PR into `main` needs at least one review from the other AI teammate.
- Keep PRs scoped to one stage — easier to review, easier to revert if needed.
