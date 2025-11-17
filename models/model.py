import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, hidden=128, n_heads=4, n_layers=2, input_dim=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        x = self.embed(x)              # (B, L, H)
        h = self.encoder(x)            # (B, L, H)
        return h.mean(dim=1)           # (B, H)


class BindingHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # required by your paper
        )
    def forward(self, h): return self.net(h)


class FunctionalHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, h): return self.net(h)


class Fusion(nn.Module):
    def __init__(self, seq_dim, motif_dim=0):
        super().__init__()
        self.fc = nn.Linear(seq_dim + motif_dim, seq_dim)

    def forward(self, seq_h, motif=None):
        if motif is None:
            return seq_h
        return self.fc(torch.cat([seq_h, motif], dim=-1))


class ProteusModel(nn.Module):
    def __init__(self, hidden=128, motif_dim=0):
        super().__init__()
        self.encoder = TransformerEncoder(hidden=hidden)
        self.fusion = Fusion(hidden, motif_dim)
        self.bind_head = BindingHead(hidden)
        self.func_head = FunctionalHead(hidden)

    def forward(self, seq, motif=None):
        h_seq = self.encoder(seq)      # (B, H)
        h = self.fusion(h_seq, motif)  # (B, H)
        return self.bind_head(h), self.func_head(h)
