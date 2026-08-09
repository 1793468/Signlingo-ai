"""
Export a trained model to ONNX for downstream mobile/on-device inference.

Owner: Youstina Wael
"""
import argparse

import torch
import torch.nn as nn

from src.models.lstm_attention import SignLSTMAttention, FEATURES


class _ONNXWrapper(nn.Module):
    """Unwrap (logits, attn_weights) -> logits only, ONNX export needs a single output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def export(checkpoint_path: str, output_path: str, num_classes: int, opset: int = 17):
    device = torch.device("cpu")

    model = SignLSTMAttention(num_classes=num_classes).to(device).float()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    wrapped = _ONNXWrapper(model)
    dummy_input = torch.randn(1, 30, FEATURES, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/sign_model.onnx")
    parser.add_argument("--num-classes", type=int, default=502)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.num_classes)
