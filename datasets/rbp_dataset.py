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
    def __init__(self, split_file, rbp_to_id=None):
        with open(split_file, "rb") as f:
            data, labels = pickle.load(f)

        self.data = data
        self.labels = labels

        # If no mapping is provided, assume single RBP → id=0
        if rbp_to_id is None:
            self.rbp_to_id = {labels[0][0]: 0}
        else:
            self.rbp_to_id = rbp_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eclip_seq = self.data[idx][0]
        rbns_seq, rbns_affinity = self.data[idx][1]

        eclip_oh = torch.tensor(one_hot_encode(eclip_seq), dtype=torch.float32)
        rbns_affinity = torch.tensor(rbns_affinity, dtype=torch.float32)

        rbp_name = self.labels[idx][0]
        rbp_id = torch.tensor(self.rbp_to_id[rbp_name], dtype=torch.long)

        expr_change = self.labels[idx][1]
        expr_tensor = None if expr_change is None else torch.tensor(expr_change, dtype=torch.float32)

        return eclip_oh, rbns_affinity, rbp_id, expr_tensor

def collate_fn(batch):
    eclip_seqs, rbns_vecs, rbp_ids, expr_changes = zip(*batch)
    return (
        torch.stack(eclip_seqs),
        torch.stack(rbns_vecs),
        torch.stack(rbp_ids),
        expr_changes,
    )

def make_loader(split_file, rbp_to_id=None, batch_size=16, shuffle=True):
    ds = RBPDataset(split_file, rbp_to_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
