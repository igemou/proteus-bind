import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def one_hot_encode(seq, alphabet="ACGU"):
    mapping = {c: i for i, c in enumerate(alphabet)}
    arr = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in mapping:
            arr[i, mapping[ch]] = 1.0
    return arr


class RBPDataset(Dataset):
    """
    One dataset for a single RBP.

    split_file: pickle with (data, bind_labels)
        - data[i] = [eclip_seq (str), (rbns_seq (str), rbns_aff (1D array-like))]
        - bind_labels[i] = 0 or 1

    pos_label_file / neg_label_file:
        - each is a pickle: (rbp_name, expr_vector)
        - we take expr_vector, average it to a scalar regression target
    """

    def __init__(self, split_file, pos_label_file, neg_label_file, rbp_id=0):
        self.rbp_id = rbp_id

        with open(split_file, "rb") as f:
            self.data, self.bind_labels = pickle.load(f)

        assert len(self.data) == len(self.bind_labels), \
            "data and bind_labels must have the same length"

        uniq = set(self.bind_labels)
        assert uniq.issubset({0, 1}), f"Non-binary binding labels detected: {uniq}"

        with open(pos_label_file, "rb") as f:
            _, pos_expr = pickle.load(f)
        with open(neg_label_file, "rb") as f:
            _, neg_expr = pickle.load(f)

        self.pos_expr_vec = torch.tensor(pos_expr, dtype=torch.float32)  # (G,)
        self.neg_expr_vec = torch.tensor(neg_expr, dtype=torch.float32)  # (G,)

        self._printed_debug = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # eCLIP + RBNS
        eclip_seq = self.data[idx][0]
        rbns_seq, rbns_aff = self.data[idx][1]

        # one-hot encode eCLIP
        seq_oh = torch.tensor(one_hot_encode(eclip_seq), dtype=torch.float32)   # (L, 4)

        # RBNS affinity → 1D tensor
        rbns_aff = torch.tensor(rbns_aff, dtype=torch.float32)                  # (D,)

        # binding label
        bind_label = torch.tensor([float(self.bind_labels[idx])], dtype=torch.float32)  # (1,)

        # functional scalar = mean log2FC of positive or negative KD vector
        func_vec = self.pos_expr_vec if bind_label.item() == 1.0 else self.neg_expr_vec
        func_scalar = func_vec.mean().unsqueeze(0)  # (1,)

        if not self._printed_debug:
            print("==== Dataset Debug ====")
            print("seq_oh:", seq_oh.shape)
            print("rbns_aff:", rbns_aff.shape)
            print("bind_label:", bind_label, bind_label.shape)
            print("func_scalar:", func_scalar, func_scalar.shape)
            print("rbp_id:", self.rbp_id)
            print("==================================")
            self._printed_debug = True

        return (
            seq_oh,                                   # (L, 4)
            rbns_aff,                                 # (D,)
            torch.tensor(self.rbp_id, dtype=torch.long),  # scalar
            bind_label,                               # (1,)
            func_scalar                               # (1,)
        )


def make_collate_fn(motif_dim):
    """
    Returns a collate_fn that:
    - pads sequences in time dimension
    - pads RBNS vectors to a fixed global motif_dim
    """

    def collate_fn(batch):
        # batch[i] = (seq_oh, rbns_aff, rbp_id, bind_label, func_scalar)
        seqs, rbns_vecs, rbp_ids, bind_labels, func_labels = zip(*batch)

        lengths = [x.size(0) for x in seqs]
        max_len = max(lengths)

        padded_seqs = []
        for seq in seqs:
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, seq.size(1))
                seq = torch.cat([seq, pad], dim=0)
            padded_seqs.append(seq)

        seqs = torch.stack(padded_seqs)  # (B, L, 4)

        padded_rbns = []
        for v in rbns_vecs:
            pad = motif_dim - v.size(0)
            if pad < 0:
                v = v[:motif_dim]
                pad = 0
            if pad > 0:
                v = torch.cat([v, torch.zeros(pad)], dim=0)
            padded_rbns.append(v)

        rbns_vecs = torch.stack(padded_rbns)  # (B, motif_dim)

        return (
            seqs,
            rbns_vecs,
            torch.stack(rbp_ids).long(),      # (B,)
            torch.stack(bind_labels),         # (B, 1)
            torch.stack(func_labels)          # (B, 1)
        )

    return collate_fn


def make_loader(split_file, pos_label_file, neg_label_file,
                rbp_id=0, batch_size=16, shuffle=True):
    """
    Simple per-RBP loader (not used in multi-RBP training, but kept for convenience).
    """
    dataset = RBPDataset(
        split_file=split_file,
        pos_label_file=pos_label_file,
        neg_label_file=neg_label_file,
        rbp_id=rbp_id
    )

    max_len = 0
    for _, (_, rbns_aff) in dataset.data:
        max_len = max(max_len, len(rbns_aff))
    collate_fn = make_collate_fn(max_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
