import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def one_hot_encode(seq, alphabet="ACGU"):
    mapping = {c: i for i, c in enumerate(alphabet)}
    arr = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in mapping:
            arr[i, mapping[ch]] = 1.0
    return arr

class RBPDataset(Dataset):
    def __init__(self, split_file, pos_label_file, neg_label_file):
        with open(split_file, "rb") as f:
            self.data, self.bind_labels = pickle.load(f)

        with open(pos_label_file, "rb") as f:
            self.pos_rbp, self.pos_expr = pickle.load(f)

        with open(neg_label_file, "rb") as f:
            self.neg_rbp, self.neg_expr = pickle.load(f)

        # Convert functional labels to tensor 
        self.pos_expr = torch.tensor(self.pos_expr, dtype=torch.float32)
        self.neg_expr = torch.tensor(self.neg_expr, dtype=torch.float32)

        self._validate_labels()
        self._printed_debug = False  

    def _validate_labels(self):
        """
        Validate binding labels from pickle.
        Must be 0/1; no other values allowed.
        """
        uniq = set(self.bind_labels)
        assert uniq.issubset({0, 1}), \
            f"ERROR: binding labels contain non-binary values: {uniq}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eclip_seq = self.data[idx][0]
        rbns_seq, rbns_aff = self.data[idx][1]

        # One-hot encode eCLIP seq
        seq_oh = torch.tensor(one_hot_encode(eclip_seq), dtype=torch.float32)

        # RBNS affinity vector
        rbns_aff = torch.tensor(rbns_aff, dtype=torch.float32)

        bind_label = float(self.bind_labels[idx])
        bind_label = torch.tensor([bind_label], dtype=torch.float32)  # shape (1,)
        func_vec = self.pos_expr if bind_label.item() == 1 else self.neg_expr
        func_scalar = func_vec.mean().unsqueeze(0)  # scalar with shape (1,)

        if not self._printed_debug:
            print("==== Dataset Debug (one-time) ====")
            print("seq_oh:", seq_oh.shape)
            print("rbns_aff:", rbns_aff.shape)
            print("bind_label:", bind_label, bind_label.shape)
            print("func_scalar:", func_scalar, func_scalar.shape)
            print("==================================")
            self._printed_debug = True

        return seq_oh, rbns_aff, bind_label, func_scalar


def collate_fn(batch):
    """
    batch[i] = (seq_oh, rbns_aff, bind_label, func_scalar)
    """
    seqs, rbns_vecs, bind_labels, func_labels = zip(*batch)

    lengths = [x.size(0) for x in seqs]
    max_len = max(lengths)

    padded = []
    for seq in seqs:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad = torch.zeros(pad_len, seq.size(1))
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)

    seqs = torch.stack(padded)                    # (B, L, 4)
    rbns_vecs = torch.stack(rbns_vecs)            # (B, D)
    bind_labels = torch.stack(bind_labels)        # (B, 1)
    func_labels = torch.stack(func_labels)        # (B, 1)

    # RBP IDs (all zeros for single-RBP setting)
    rbp_ids = torch.zeros(len(batch), dtype=torch.long)

    assert bind_labels.ndim == 2 and bind_labels.shape[1] == 1, \
        f"bind_labels wrong shape: {bind_labels.shape}"

    assert torch.all(bind_labels >= 0) and torch.all(bind_labels <= 1), \
        f"Invalid bind labels: {bind_labels}"

    assert func_labels.ndim == 2 and func_labels.shape[1] == 1, \
        f"func_labels wrong shape: {func_labels.shape}"

    return seqs, rbns_vecs, rbp_ids, func_labels, bind_labels


def make_loader(split, pos, neg, batch_size=16, shuffle=True):
    ds = RBPDataset(split, pos, neg)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn)
