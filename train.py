import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from models.model import ProteusModel
from datasets.rbp_dataset import RBPDataset, make_collate_fn
from utils.losses import multitask_loss
from utils.metrics import binding_metrics, functional_metrics

def train_one_epoch(model, loader, optim, device,
                    lambda_bind=1.0, lambda_func=0.5):

    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    # batch: (seqs, rbns_vecs, rbp_ids, bind_labels, func_labels)
    for seq, rbns_vec, rbp_ids, bind_label, func_label in pbar:
        seq = seq.to(device)          # (B, L, 4)
        rbns_vec = rbns_vec.to(device)  # (B, motif_dim)
        rbp_ids = rbp_ids.to(device)    # (B,)
        bind_label = bind_label.to(device)  # (B, 1)
        func_label = func_label.to(device)  # (B, 1)

        pred_b, pred_f = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        loss, Lb, Lf = multitask_loss(
            pred_b, bind_label,
            pred_f, func_label,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_true_b, all_pred_b = [], []
    all_true_f, all_pred_f = [], []

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

    yb = torch.cat(all_true_b)   # (N, 1)
    pb = torch.cat(all_pred_b)   # (N, 1)
    yf = torch.cat(all_true_f)   # (N, 1)
    pf = torch.cat(all_pred_f)   # (N, 1)

    results = {}
    results.update(binding_metrics(yb, pb))
    results.update(functional_metrics(yf, pf))
    return results

def build_datasets(split_files, pos_label_files, neg_label_files):
    """
    Create a list of RBPDataset instances, one per RBP.
    rbp_id is the index in the lists.
    """
    datasets = []
    for rbp_id, (split, pos, neg) in enumerate(
        zip(split_files, pos_label_files, neg_label_files)
    ):
        ds = RBPDataset(
            split_file=split,
            pos_label_file=pos,
            neg_label_file=neg,
            rbp_id=rbp_id
        )
        datasets.append(ds)
    return datasets


def compute_global_motif_dim(datasets):
    """
    Scan ALL datasets to find the maximum RBNS affinity length.
    This defines a single global motif_dim used by the model and collate_fn.
    """
    max_len = 0
    for ds in datasets:
        for _, (_, rbns_aff) in ds.data:
            try:
                L = len(rbns_aff)
            except TypeError:
                continue
            if L > max_len:
                max_len = L
    return max_len


def make_loader_from_datasets(datasets, batch_size, shuffle, motif_dim):
    """
    Build a DataLoader from a list of RBPDataset objects, using ConcatDataset
    and a collate_fn that pads RBNS vectors to motif_dim.
    """
    if len(datasets) == 1:
        full_ds = datasets[0]
    else:
        full_ds = ConcatDataset(datasets)

    collate_fn = make_collate_fn(motif_dim)

    loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return loader

def train(train_files, val_files,
          pos_label_files, neg_label_files,
          save_path="best_model.pt",
          epochs=20,
          batch_size=16, patience=7,
          lr=1e-4,
          lambda_bind=1.0, lambda_func=0.5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build per-RBP datasets for train/val
    train_datasets = build_datasets(train_files, pos_label_files, neg_label_files)
    val_datasets = build_datasets(val_files, pos_label_files, neg_label_files)

    # Compute global motif_dim across ALL RBPs and splits
    all_datasets = train_datasets + val_datasets
    motif_dim = compute_global_motif_dim(all_datasets)
    num_rbps = len(train_datasets)

    print(f"> num_rbps = {num_rbps}")
    print(f"> global motif_dim = {motif_dim}")

    # Build loaders
    train_loader = make_loader_from_datasets(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        motif_dim=motif_dim
    )
    val_loader = make_loader_from_datasets(
        val_datasets,
        batch_size=batch_size,
        shuffle=False,
        motif_dim=motif_dim
    )

    # Model
    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=num_rbps,
        rbp_emb_dim=32
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = -float("inf")
    patience_counter = 0

    print("\n=== Starting Training ===\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optim, device,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func
        )

        val_scores = evaluate(model, val_loader, device)
        val_metric = val_scores.get("AUROC", 0.0)

        print(f"Epoch {epoch:02d} | "
              f"Loss={train_loss:.4f} | AUROC={val_metric:.4f} | "
              f"Scores={val_scores}")

        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "val_scores": val_scores,
            }, save_path)

            print(f"Saved best model → {save_path}\n")
        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining finished.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True,
                        help="Comma-separated train split paths")
    parser.add_argument("--val", required=True,
                        help="Comma-separated val split paths")
    parser.add_argument("--pos_label", required=True,
                        help="Comma-separated pos label paths")
    parser.add_argument("--neg_label", required=True,
                        help="Comma-separated neg label paths")

    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lambda_bind", type=float, default=1.0)
    parser.add_argument("--lambda_func", type=float, default=0.5)

    args = parser.parse_args()

    train_files = [x.strip() for x in args.train.split(",")]
    val_files = [x.strip() for x in args.val.split(",")]
    pos_label_files = [x.strip() for x in args.pos_label.split(",")]
    neg_label_files = [x.strip() for x in args.neg_label.split(",")]

    n = len(train_files)
    assert len(val_files) == n and len(pos_label_files) == n and len(neg_label_files) == n, \
        "All argument lists must have same number of comma-separated items"

    train(
        train_files=train_files,
        val_files=val_files,
        pos_label_files=pos_label_files,
        neg_label_files=neg_label_files,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        lr=args.lr,
        lambda_bind=args.lambda_bind,
        lambda_func=args.lambda_func,
    )
