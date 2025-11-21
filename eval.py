import torch
from torch.utils.data import DataLoader, ConcatDataset

from models.model import ProteusModel
from datasets.rbp_dataset import RBPDataset, make_collate_fn
from utils.metrics import (
    binding_metrics,
    functional_metrics,
    next_base_metrics,   
)


@torch.no_grad()
def evaluate(model, loader, device="cpu", has_next_head=True):
    """
    Evaluate a saved model on a test loader.
    Supports:
      - binding prediction
      - functional prediction
      - next-base prediction (optional)
    """

    model.eval()

    all_true_b, all_pred_b = [], []
    all_true_f, all_pred_f = [], []

    all_true_next, all_pred_next = [], [] if has_next_head else (None, None)

    for seq, rbns_vec, rbp_ids, bind_label, func_label, next_label in loader:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)
        bind_label = bind_label.to(device)
        func_label = func_label.to(device)
        next_label = next_label.to(device)
        out = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        if has_next_head:
            pred_b, pred_f, pred_next = out
        else:
            pred_b, pred_f = out
            pred_next = None

        # store binding
        all_true_b.append(bind_label.cpu())
        all_pred_b.append(pred_b.cpu())

        # store functional
        all_true_f.append(func_label.cpu())
        all_pred_f.append(pred_f.cpu())

        # store next-base 
        if has_next_head:
            all_true_next.append(next_label.cpu())
            all_pred_next.append(pred_next.cpu())

    # concat
    yb = torch.cat(all_true_b)
    pb = torch.cat(all_pred_b)
    yf = torch.cat(all_true_f)
    pf = torch.cat(all_pred_f)

    results = {}
    results.update(binding_metrics(yb, pb))
    results.update(functional_metrics(yf, pf))

    if has_next_head:
        yn = torch.cat(all_true_next)
        pn = torch.cat(all_pred_next)
        results.update(next_base_metrics(yn, pn))

    return results


def build_dataset_for_eval(split_file, pos_file, neg_file, rbp_id):
    """
    A lightweight wrapper so evaluation does NOT try to construct
    single-RBP cross-negative mode.
    We use multi_RBP=True during evaluation always.
    """

    return RBPDataset(
        all_split_files=[split_file],
        all_pos_label_files=[pos_file],
        all_neg_label_files=[neg_file],
        target_rbp_id=None,
        multi_RBP=True,
    )


def build_eval_datasets(test_files, pos_files, neg_files):
    datasets = []
    for rbp_id, (split, pos, neg) in enumerate(zip(test_files, pos_files, neg_files)):
        ds = build_dataset_for_eval(split, pos, neg, rbp_id)
        datasets.append(ds)
    return datasets


def compute_motif_dim(datasets):
    max_len = 0
    for ds in datasets:
        for seq_item in ds.data:
            _, (_, rbns_aff) = seq_item
            if len(rbns_aff) > max_len:
                max_len = len(rbns_aff)
    return max_len


def make_eval_loader(datasets, batch_size, motif_dim):
    if len(datasets) == 1:
        full_ds = datasets[0]
    else:
        full_ds = ConcatDataset(datasets)

    collate_fn = make_collate_fn(motif_dim)

    return DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--pos_label", required=True)
    parser.add_argument("--neg_label", required=True)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_files = [s.strip() for s in args.test.split(",")]
    pos_files = [s.strip() for s in args.pos_label.split(",")]
    neg_files = [s.strip() for s in args.neg_label.split(",")]

    eval_datasets = build_eval_datasets(test_files, pos_files, neg_files)

    motif_dim = compute_motif_dim(eval_datasets)
    num_rbps = len(eval_datasets)

    print(f"> motif_dim = {motif_dim}")
    print(f"> num_rbps = {num_rbps}")

    eval_loader = make_eval_loader(eval_datasets, args.batch, motif_dim)
    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=num_rbps,
        rbp_emb_dim=32,
    ).to(device)

    print(f"Loading checkpoint: {args.model}")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    has_next_head = "next_head" in model.__dict__ or \
                    hasattr(model, "next_head") or \
                    any("next" in k for k in model.state_dict().keys())

    print(f"> Model has next-base head? {has_next_head}")

    print("\nRunning evaluation...\n")
    results = evaluate(model, eval_loader, device, has_next_head=has_next_head)

    print("====== FINAL TEST METRICS ======")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
