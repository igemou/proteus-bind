import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def one_hot_encode(seq, alphabet="ACGU"):
    mapping = {c: i for i, c in enumerate(alphabet)}
    arr = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in mapping:
            arr[i, mapping[ch]] = 1.0
    return arr

class RBPDataset(Dataset):
    """
    Unified dataset loader.
      - MODE 1: single-RBP training with cross-RBP negatives (multi_RBP=False)
      - MODE 2: multi-RBP training (multi_RBP=True)

    if multi_RBP = True  (MODE 2):
        - For each RBP:
            data: mixed pos+neg examples from its split file
        - y_bind: original bind label (0/1)
        - y_func: per-RBP functional vector
          (pos_vec for y=1, neg_vec for y=0)
        - rbp_id: identifies which RBP this example belongs to

    else multi_RBP = False (MODE 1, single-RBP):
        - target_rbp_id = index of RBP we are training on (A)
        - positives = all positive examples (y=1) from target RBP A
        - negatives = positive examples (y=1) from ALL OTHER RBPs (B, C, ...)
                      relabeled as negative (0) for A
        - All examples use rbp_id = target_rbp_id so the model
          is “conditioning” on RBP A only.
    """

    def __init__(
        self,
        all_split_files,
        all_pos_label_files,
        all_neg_label_files,
        target_rbp_id=None,
        multi_RBP=True,
    ):
        self.multi_RBP = multi_RBP
        self.target_rbp_id = target_rbp_id
        self.all_data = []

        for rbp_id, (split_file, pos_file, neg_file) in enumerate(
            zip(all_split_files, all_pos_label_files, all_neg_label_files)
        ):
            with open(split_file, "rb") as f:
                data, labels = pickle.load(f)

            with open(pos_file, "rb") as f:
                _, pos_vec = pickle.load(f)
            with open(neg_file, "rb") as f:
                _, neg_vec = pickle.load(f)

            pos_vec = torch.tensor(pos_vec, dtype=torch.float32)
            neg_vec = torch.tensor(neg_vec, dtype=torch.float32)

            # data: list of (eclip_seq, (rbns_seq, rbns_aff))
            # labels: list/array of bind labels (0/1)
            self.all_data.append((data, labels, pos_vec, neg_vec, rbp_id))

        if self.multi_RBP:
            self.data = []
            self.labels = []
            self.rbp_ids = []
            self.func_vecs = []

            for (data, labels, pos_vec, neg_vec, rbp_id) in self.all_data:
                for seq_item, y in zip(data, labels):
                    self.data.append(seq_item)
                    self.labels.append(float(y))
                    self.rbp_ids.append(rbp_id)
                    self.func_vecs.append(pos_vec if y == 1 else neg_vec)

            return  

        assert target_rbp_id is not None, "Must specify target_rbp_id in single-RBP mode"

        # Take the target RBP A
        A_data, A_labels, A_pos_vec, A_neg_vec, _ = self.all_data[target_rbp_id]

        # positives = A_pos
        positives = [
            (item, 1, target_rbp_id, A_pos_vec)
            for item, y in zip(A_data, A_labels)
            if y == 1
        ]

        # negatives = positives from other RBPs
        negatives = []
        for (data, labels, pos_vec, neg_vec, rbp_id) in self.all_data:
            if rbp_id == target_rbp_id:
                continue
            for item, y in zip(data, labels):
                if y == 1:  # positive from RBP_B etc.
                    negatives.append((item, 0, target_rbp_id, A_neg_vec))

        print(
            f"[Single-RBP mode] RBP {target_rbp_id}: "
            f"{len(positives)} positives, {len(negatives)} negatives (from other RBPs)."
        )

        merged = positives + negatives

        self.data = [x[0] for x in merged]   # seq_item
        self.labels = [x[1] for x in merged]
        self.rbp_ids = [x[2] for x in merged]
        self.func_vecs = [x[3] for x in merged]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eclip_seq = self.data[idx][0]

        # convert DNA → RNA
        eclip_seq = eclip_seq.replace("T", "U")

        rbns_seq, rbns_aff = self.data[idx][1]

        seq_oh = torch.tensor(one_hot_encode(eclip_seq), dtype=torch.float32)
        rbns_oh = torch.tensor(one_hot_encode(rbns_seq), dtype=torch.float32)
        rbns_aff = torch.tensor(rbns_aff, dtype=torch.float32)

        y_bind = torch.tensor([self.labels[idx]], dtype=torch.float32)
        y_func = self.func_vecs[idx].mean().unsqueeze(0)

        # Next-token (now safe)
        alphabet = "ACGU"
        base_to_idx = {c: i for i, c in enumerate(alphabet)}
        last_base = eclip_seq[-1]
        y_next = torch.tensor(base_to_idx[last_base], dtype=torch.long)

        return (
            seq_oh, rbns_oh, rbns_aff,
            torch.tensor(self.rbp_ids[idx], dtype=torch.long),
            y_bind, y_func, y_next
        )




def make_collate_fn(motif_dim: int):
    """
    Returns a collate_fn that:
      - pads one-hot RNA sequences in a batch to the same length
      - pads/truncates RBNS affinity vectors to motif_dim
      - stacks binding, functional, next-base labels
    """

    def collate(batch):
        seqs, rbns, rbns_vecs, rbp_ids, y_binds, y_funcs, y_nexts = zip(*batch)

        batch_size = len(seqs)
        max_len = max(s.shape[0] for s in seqs)
        seq_width = seqs[0].shape[1]  # should be 4

        # Pad sequences
        seq_batch = torch.zeros(batch_size, max_len, seq_width, dtype=torch.float32)
        for i, s in enumerate(seqs):
            L = s.shape[0]
            seq_batch[i, :L, :] = s
            
        # Pad RBNS sequences
        rbns_seq_batch = torch.zeros(batch_size, max_len, seq_width, dtype=torch.float32)
        for i, s in enumerate(rbns):
            L = s.shape[0]
            rbns_seq_batch[i, :L, :] = s

        # Pad RBNS vectors
        rbns_aff_batch = torch.zeros(batch_size, motif_dim, dtype=torch.float32)
        for i, v in enumerate(rbns_vecs):
            L = min(len(v), motif_dim)
            rbns_aff_batch[i, :L] = v[:L]

        rbp_ids_tensor = torch.stack(rbp_ids, dim=0)     # (B,)
        y_bind_batch = torch.stack(y_binds, dim=0)       # (B,1)
        y_func_batch = torch.stack(y_funcs, dim=0)       # (B,1)
        y_next_batch = torch.stack(y_nexts, dim=0)       # (B,)

        return (
            seq_batch,
            rbns_seq_batch,
            rbns_aff_batch,
            rbp_ids_tensor,
            y_bind_batch,
            y_func_batch,
            y_next_batch,
        )

    return collate
