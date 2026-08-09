# SignLingo — AI Subsystem

AI models for **SignLingo**, an AI/XR/embedded graduation project for deaf and mute
communication support. This repo contains the two AI pipelines owned by the AI sub-team:

| Pipeline | Description | Status |
|---|---|---|
| [`sign-to-text/`](./sign-to-text) | Arabic Sign Language (KARSL-502) → text, via MediaPipe landmarks + BiLSTM-Attention | In progress |
| `speech-to-text/` | Egyptian Arabic speech → text, via Whisper + LoRA | In progress |

This is one of several repos that make up the full SignLingo system
(Mobile — Flutter, Backend — Laravel, VR — Unity/Meta Quest, AR — Unity, AI — this repo).

## Team (AI)

| Member | Responsibility |
|---|---|
| **Mariam Ashraf Tobar** | Data preprocessing pipeline, BiLSTM + Attention model, training & evaluation |
| **Youstina Wael** | Code review, ONNX export & inference optimization, backend integration, test cases |

## Repo layout

```
signlingo-ai/
├── sign-to-text/     # ArSL video → text
├── speech-to-text/   # Egyptian Arabic audio → text (added next)
└── docs/             # Shared reports, diagrams
```

Each pipeline is self-contained (own README, requirements, configs) so it can be
developed, tested, and versioned independently.

## Getting started

See [`sign-to-text/README.md`](./sign-to-text/README.md) for setup and usage.

## License

MIT — see [LICENSE](./LICENSE).
