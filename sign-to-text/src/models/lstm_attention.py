"""
BiLSTM + temporal-attention model for sign classification.

Owner: Mariam Ashraf Tobar
Test accuracy: 97.72% (120-sign subset)
"""
import torch
import torch.nn as nn

FEATURES = 126


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (B, T, 2H)
        scores = self.attn(lstm_out).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # (B, 2H)
        return context, weights


class SignLSTMAttention(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.fc(context)
        return logits, attn_weights
