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
        x = self.embed(x)          # (B, L, H)
        h = self.encoder(x)        # (B, L, H)
        return h.mean(dim=1)       # (B, H)


class Fusion(nn.Module):
    def __init__(self, dim_in, motif_dim=0):
        super().__init__()
        self.fc = nn.Linear(dim_in + motif_dim, dim_in)

    def forward(self, h, motif=None):
        if motif is None:
            return h
        return self.fc(torch.cat([h, motif], dim=-1))


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


class ProteusModel(nn.Module):
    """
    Multi-RBP model with:
        - Transformer encoder (eCLIP)
        - RBNS fusion
        - RBP embedding (optional)
        - Binding + functional heads
    """

    def __init__(self, hidden=128, motif_dim=0, num_rbps=1, rbp_emb_dim=32):
        super().__init__()

        self.encoder = TransformerEncoder(hidden=hidden)
        self.rbp_embed = nn.Embedding(num_rbps, rbp_emb_dim)

        # Encoder output + RBP embedding → fused backbone
        self.fusion = Fusion(hidden + rbp_emb_dim, motif_dim)

        self.bind_head = BindingHead(hidden)
        self.func_head = FunctionalHead(hidden)

    def forward(self, seq, motif=None, rbp_id=None):
        """
        seq:        (B, L, 4)
        motif:      (B, D) or None
        rbp_id:     (B,) long tensor
        """

        h_seq = self.encoder(seq)               # (B, H)

        if rbp_id is None:
            # Single-RBP mode: use embedding for RBP 0
            rbp_id = torch.zeros(seq.size(0), dtype=torch.long, device=seq.device)

        h_rbp = self.rbp_embed(rbp_id)          # (B, rbp_emb_dim)

        h_joint = torch.cat([h_seq, h_rbp], dim=-1)   # (B, H + rbp_emb_dim)

        h = self.fusion(h_joint, motif)         # (B, H)

        pred_b = self.bind_head(h)              # (B, 1)
        pred_f = self.func_head(h)              # (B, 1)

        return pred_b, pred_f
