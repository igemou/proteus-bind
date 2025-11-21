import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, hidden=128, n_heads=4, n_layers=2, input_dim=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        x = self.embed(x)
        h = self.encoder(x)
        return h.mean(dim=1)


class Fusion(nn.Module):
    def __init__(self, hidden, rbp_emb_dim, motif_dim=0):
        super().__init__()
        self.fc = nn.Linear(hidden + rbp_emb_dim + motif_dim, hidden)

    def forward(self, h_seq, h_rbp, motif=None):
        if motif is None:
            x = torch.cat([h_seq, h_rbp], dim=-1)
        else:
            x = torch.cat([h_seq, h_rbp, motif], dim=-1)
        return self.fc(x)


class BindingHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        return self.net(h)


class FunctionalHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h):
        return self.net(h)

class NextBaseHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 nucleotide classes A, C, G, U
        )

    def forward(self, h):
        return self.net(h)   # raw logits

class ProteusModel(nn.Module):
    def __init__(self, hidden=128, motif_dim=0, num_rbps=1, rbp_emb_dim=32):
        super().__init__()

        self.encoder = TransformerEncoder(hidden=hidden)
        self.rbp_embed = nn.Embedding(num_rbps, rbp_emb_dim)

        self.fusion = Fusion(hidden, rbp_emb_dim, motif_dim)
        self.bind_head = BindingHead(hidden)
        self.func_head = FunctionalHead(hidden)
        self.next_base_head = NextBaseHead(hidden)

    def forward(self, seq, motif=None, rbp_id=None):
        h_seq = self.encoder(seq)

        # default: single RBP
        if rbp_id is None:
            rbp_id = torch.zeros(seq.size(0), dtype=torch.long, device=seq.device)

        h_rbp = self.rbp_embed(rbp_id)

        h = self.fusion(h_seq, h_rbp, motif)

        bind_pred = self.bind_head(h)
        func_pred = self.func_head(h)
        next_base_logits = self.next_base_head(h)

        return bind_pred, func_pred, next_base_logits
