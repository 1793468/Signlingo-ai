"""
Export the merged Whisper model to ONNX (encoder + decoder + decoder-with-past),
the same integration path sign-to-text used for backend/inference optimization.

Requires the merged model from src/merge_and_export.py — export a raw
base+LoRA pair directly isn't supported by optimum's exporter.

Usage:
    python src/export_onnx.py \
        --model_dir speech-to-text/checkpoints/merged \
        --out_dir speech-to-text/checkpoints/onnx
"""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    # optimum-cli handles the multi-part whisper export (encoder/decoder/
    # decoder_with_past) and writes a config the ORT runtime understands.
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", args.model_dir,
        "--task", "automatic-speech-recognition",
        args.out_dir,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"ONNX model exported to {args.out_dir}")
    print(
        "Load it from the backend with optimum.onnxruntime.ORTModelForSpeechSeq2Seq, "
        "or point a Whisper-compatible ONNX runtime (e.g. whisper.cpp, ctranslate2 via "
        "a separate conversion) at these files depending on what the Laravel backend "
        "calls into."
    )


if __name__ == "__main__":
    main()
