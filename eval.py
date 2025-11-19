import torch
from torch.utils.data import DataLoader, ConcatDataset

from models.model import ProteusModel
from datasets.rbp_dataset import RBPDataset, make_collate_fn
from utils.metrics import binding_metrics, functional_metrics

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()

    all_true_b, all_pred_b = [], []
    all_true_f, all_pred_f = [], []

    # batch: (seq, rbns_vec, rbp_ids, bind_label, func_label)
    for seq, rbns_vec, rbp_ids, bind_label, func_label in loader:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)
        bind_label = bind_label.to(device)
        func_label = func_label.to(device)

        pred_b, pred_f = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        all_true_b.append(bind_label.cpu())
        all_pred_b.append(pred_b.cpu())

        all_true_f.append(func_label.cpu())
        all_pred_f.append(pred_f.cpu())

    yb = torch.cat(all_true_b)
    pb = torch.cat(all_pred_b)
    yf = torch.cat(all_true_f)
    pf = torch.cat(all_pred_f)

    results = {}
    results.update(binding_metrics(yb, pb))
    results.update(functional_metrics(yf, pf))

    return results


def build_datasets(test_files, pos_files, neg_files):
    datasets = []
    for rbp_id, (split, pos, neg) in enumerate(zip(test_files, pos_files, neg_files)):
        ds = RBPDataset(
            split_file=split,
            pos_label_file=pos,
            neg_label_file=neg,
            rbp_id=rbp_id
        )
        datasets.append(ds)
    return datasets


def compute_motif_dim(datasets):
    """Global motif_dim across ALL RBPs (same as training)."""
    max_len = 0
    for ds in datasets:
        for _, (_, rbns_aff) in ds.data:
            L = len(rbns_aff)
            if L > max_len:
                max_len = L
    return max_len


def make_loader_from_datasets(datasets, batch_size, motif_dim):
    """ConcatDataset + correct collate_fn."""
    if len(datasets) == 1:
        full_ds = datasets[0]
    else:
        full_ds = ConcatDataset(datasets)

    collate_fn = make_collate_fn(motif_dim)

    return DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True, help="Comma-separated test .pkl files")
    parser.add_argument("--pos_label", required=True, help="Comma-separated pos label .pkl files")
    parser.add_argument("--neg_label", required=True, help="Comma-separated neg label .pkl files")
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_files = [s.strip() for s in args.test.split(",")]
    pos_files = [s.strip() for s in args.pos_label.split(",")]
    neg_files = [s.strip() for s in args.neg_label.split(",")]

    # Build datasets
    test_datasets = build_datasets(test_files, pos_files, neg_files)

    # Compute global motif_dim & num_rbps (must match training)
    motif_dim = compute_motif_dim(test_datasets)
    num_rbps = len(test_datasets)

    print(f"> motif_dim = {motif_dim}")
    print(f"> num_rbps = {num_rbps}")

    # Build loader
    test_loader = make_loader_from_datasets(
        test_datasets,
        batch_size=args.batch,
        motif_dim=motif_dim
    )

    # Build model
    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=num_rbps,
        rbp_emb_dim=32
    ).to(device)

    print(f"Loading checkpoint from {args.model}...")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Evaluate
    print("\nRunning evaluation...\n")
    results = evaluate(model, test_loader, device)

    print("====== FINAL TEST METRICS ======")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
