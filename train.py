import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.model import ProteusModel
from datasets.rbp_dataset import RBPDataset, make_collate_fn
from utils.losses import multitask_loss
from utils.metrics import binding_metrics, functional_metrics, next_base_metrics

def train_one_epoch(
    model,
    loader,
    optim,
    device,
    lambda_bind: float = 1.0,
    lambda_func: float = 0.5,
    lambda_next: float = 0.0,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for seq, rbns_vec, rbp_ids, bind_label, func_label, next_label in pbar:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)
        bind_label = bind_label.to(device)
        func_label = func_label.to(device)
        next_label = next_label.to(device)

        # Forward: model returns binding, functional, and next-base logits
        pred_b, pred_f, pred_next = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        # Multitask loss (binding + optional functional + optional next-base)
        loss, Lb, Lf, Ln = multitask_loss(
            pred_b,
            bind_label,
            pred_f,
            func_label,
            pred_next,
            next_label,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func,
            lambda_next=lambda_next,
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
    all_true_next, all_pred_next = [], []

    for seq, rbns_vec, rbp_ids, bind_label, func_label, next_label in loader:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)
        bind_label = bind_label.to(device)
        func_label = func_label.to(device)
        next_label = next_label.to(device)

        pred_b, pred_f, pred_next = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        all_true_b.append(bind_label.cpu())
        all_pred_b.append(pred_b.cpu())

        all_true_f.append(func_label.cpu())
        all_pred_f.append(pred_f.cpu())

        all_true_next.append(next_label.cpu())
        all_pred_next.append(pred_next.cpu())

    yb  = torch.cat(all_true_b)
    pb  = torch.cat(all_pred_b)
    yf  = torch.cat(all_true_f)
    pf  = torch.cat(all_pred_f)
    yn  = torch.cat(all_true_next)
    pn  = torch.cat(all_pred_next)

    results = {}
    results.update(binding_metrics(yb, pb))
    results.update(functional_metrics(yf, pf))
    results.update(next_base_metrics(yn, pn))  # <-- NEW

    return results

def compute_motif_dim(datasets):
    max_len = 0
    for ds in datasets:
        for seq_item in ds.data:
            # seq_item is (eclip_seq, (rbns_seq, rbns_aff))
            _, (_, rbns_aff) = seq_item
            try:
                L = len(rbns_aff)
            except TypeError:
                continue
            if L > max_len:
                max_len = L
    return max_len


def train(
    train_files,
    val_files,
    pos_label_files,
    neg_label_files,
    save_path="best_model.pt",
    epochs=20,
    batch_size=16,
    patience=7,
    lr=1e-4,
    lambda_bind=1.0,
    lambda_func=0.5,
    lambda_next=0.0,
    mode="multi",
    target_rbp_id=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_rbps = len(train_files)

    if mode not in ("multi", "single"):
        raise ValueError(f"Unknown mode: {mode}. Use 'multi' or 'single'.")

    is_multi = mode == "multi"

    if not is_multi:
        if target_rbp_id is None:
            raise ValueError("In single-RBP mode you must pass target_rbp_id.")
        if not (0 <= target_rbp_id < num_rbps):
            raise ValueError(
                f"target_rbp_id must be in [0, {num_rbps - 1}], got {target_rbp_id}."
            )

    mode_label = "multi-RBP" if is_multi else f"single-RBP (target={target_rbp_id})"
    print(f"\n=== Config ===")
    print(f"  mode         = {mode_label}")
    print(f"  num_rbps     = {num_rbps}")
    print(f"  save_path    = {save_path}")
    print(f"  epochs       = {epochs}")
    print(f"  batch_size   = {batch_size}")
    print(f"  lr           = {lr}")
    print(f"  λ_bind       = {lambda_bind}")
    print(f"  λ_func       = {lambda_func}")
    print(f"  λ_next       = {lambda_next}")
    print("==============\n")

    # ----- Build datasets -----
    train_dataset = RBPDataset(
        all_split_files=train_files,
        all_pos_label_files=pos_label_files,
        all_neg_label_files=neg_label_files,
        target_rbp_id=target_rbp_id,
        multi_RBP=is_multi,
    )

    val_dataset = RBPDataset(
        all_split_files=val_files,
        all_pos_label_files=pos_label_files,
        all_neg_label_files=neg_label_files,
        target_rbp_id=target_rbp_id,
        multi_RBP=is_multi,
    )

    # motif_dim based on both train+val
    motif_dim = compute_motif_dim([train_dataset, val_dataset])
    print(f"> global motif_dim = {motif_dim}")

    collate_fn = make_collate_fn(motif_dim)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # num_rbps = total RBPs (consistent in both modes)
    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=num_rbps,
        rbp_emb_dim=32,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = -float("inf")
    patience_counter = 0

    print(f"\n=== Starting Training ({mode_label}) ===\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func,
            lambda_next=lambda_next,
        )

        val_scores = evaluate(model, val_loader, device)
        val_metric = val_scores.get("AUROC", 0.0)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss={train_loss:.4f} | AUROC={val_metric:.4f} | "
            f"Scores={val_scores}"
        )

        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "val_scores": val_scores,
                    "mode": mode,
                    "target_rbp_id": target_rbp_id,
                    "lambda_bind": lambda_bind,
                    "lambda_func": lambda_func,
                    "lambda_next": lambda_next,
                },
                save_path,
            )
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

    parser.add_argument(
        "--train",
        required=True,
        help="Comma-separated train split paths (one per RBP)",
    )
    parser.add_argument(
        "--val",
        required=True,
        help="Comma-separated val split paths (one per RBP)",
    )
    parser.add_argument(
        "--pos_label",
        required=True,
        help="Comma-separated pos functional label paths (one per RBP)",
    )
    parser.add_argument(
        "--neg_label",
        required=True,
        help="Comma-separated neg functional label paths (one per RBP)",
    )

    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lambda_bind", type=float, default=1.0)
    parser.add_argument("--lambda_func", type=float, default=0.5)
    parser.add_argument(
        "--lambda_next",
        type=float,
        default=0.0,
        help="Weight for next-base prediction loss (0.0 disables it).",
    )
    parser.add_argument(
        "--mode",
        choices=["multi", "single"],
        default="multi",
        help="Training mode: 'multi' = multi-RBP, 'single' = single-RBP with cross-RBP negatives.",
    )
    parser.add_argument(
        "--target_rbp",
        type=int,
        default=None,
        help="Index of target RBP (0..N-1) when --mode single.",
    )

    args = parser.parse_args()

    train_files = [x.strip() for x in args.train.split(",")]
    val_files = [x.strip() for x in args.val.split(",")]
    pos_label_files = [x.strip() for x in args.pos_label.split(",")]
    neg_label_files = [x.strip() for x in args.neg_label.split(",")]

    n = len(train_files)
    assert (
        len(val_files) == n
        and len(pos_label_files) == n
        and len(neg_label_files) == n
    ), "All argument lists must have same number of comma-separated items"

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
        lambda_next=args.lambda_next,
        mode=args.mode,
        target_rbp_id=args.target_rbp,
    )
