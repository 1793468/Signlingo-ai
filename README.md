# SignLingo — AI Subsystem

AI models for **SignLingo**, an AI/XR/embedded graduation project for deaf and mute
communication support. This repo contains the two AI pipelines owned by the AI sub-team:

| Pipeline | Description | Status |
|---|---|---|
| [`sign-to-text/`](./sign-to-text) | Arabic Sign Language (KARSL-502) → text, via MediaPipe landmarks + BiLSTM-Attention | Finished and deployed|
| [`speech-to-text/`](./speech-to-text) | Egyptian Arabic speech → text, via Whisper + LoRA | complete - future improvements planned |

This is one of several repos that make up the full SignLingo system
(Mobile — Flutter, Backend — Laravel, VR — Unity/Meta Quest, AR — Unity, AI — this repo).

## Team (AI)

Each pipeline has its own team breakdown, since the two are built and reviewed
independently — see [`sign-to-text/README.md`](./sign-to-text/README.md#workflow--handoff)
and [`speech-to-text/README.md`](./speech-to-text/README.md#team) for who owns what on
each.

## Repo layout

```
signlingo-ai/
├── sign-to-text/     # ArSL video → text
├── speech-to-text/   # Egyptian Arabic audio → text
└── docs/             # Shared reports, diagrams
```

Each pipeline is self-contained (own README, requirements, configs) so it can be
developed, tested, and versioned independently.

## Getting started

See [`sign-to-text/README.md`](./sign-to-text/README.md) or
[`speech-to-text/README.md`](./speech-to-text/README.md) for setup and usage.

## License

MIT — see [LICENSE](./LICENSE).
