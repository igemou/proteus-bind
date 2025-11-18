import os
import torch
from torch.utils.data import DataLoader

from models.model import ProteusModel
from datasets.rbp_dataset import make_loader
from utils.losses import multitask_loss
from utils.metrics import binding_metrics, functional_metrics


def train_one_epoch(model, loader, optim, device, lambda_bind=1.0, lambda_func=0.5):
    model.train()
    total_loss = 0.0

    for seq, rbns_vec, rbp_ids, expr_changes in loader:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)

        # Binding labels = 1 for all eCLIP peaks
        true_bind = torch.ones(seq.size(0), 1, device=device)

        # Build functional targets + mask
        func_list = []
        mask_list = []

        for x in expr_changes:
            if x is None:
                func_list.append(torch.zeros(1))  # placeholder
                mask_list.append(0)
            else:
                func_list.append(torch.tensor(x, dtype=torch.float32))
                mask_list.append(1)

        true_func = torch.stack(func_list).to(device)  # (B,1)
        mask = torch.tensor(mask_list, device=device).float()

        pred_b, pred_f = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        loss, L_bind, L_func = multitask_loss(
            pred_b, true_bind,
            pred_f, true_func,
            mask,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()

    all_true_b, all_pred_b = [], []
    all_true_f, all_pred_f = [], []

    with torch.no_grad():
        for seq, rbns_vec, rbp_ids, expr_changes in loader:

            seq = seq.to(device)
            rbns_vec = rbns_vec.to(device)
            rbp_ids = rbp_ids.to(device)

            pred_b, pred_f = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

            true_bind = torch.ones(seq.size(0), 1, device=device)
            all_true_b.append(true_bind.cpu())
            all_pred_b.append(pred_b.cpu())

            for p, y in zip(pred_f.cpu(), expr_changes):
                if y is not None:
                    all_pred_f.append(p)
                    all_true_f.append(torch.tensor(y))

    results = {}

    if len(all_true_b) > 0:
        yb = torch.cat(all_true_b)
        pb = torch.cat(all_pred_b)
        results.update(binding_metrics(yb, pb))

    if len(all_true_f) > 0:
        yf = torch.stack(all_true_f)
        pf = torch.stack(all_pred_f)
        results.update(functional_metrics(yf, pf))

    return results

def train(train_file, val_file, save_path="best_model.pt", num_rbps=1, epochs=30, 
          batch_size=16, patience=7, lr=1e-4, lambda_bind=1.0, lambda_func=0.5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loaders
    train_loader = make_loader(train_file, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_file, batch_size=batch_size, shuffle=False)

    model = ProteusModel(
        hidden=128,
        motif_dim=train_loader.dataset.data[0][1][1].shape[0],
        num_rbps=num_rbps,
        rbp_emb_dim=32
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = -float("inf")
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optim, device,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func
        )

        val_scores = evaluate(model, val_loader, device)

        # Early stopping metric = AUROC (or change this to Spearman)
        val_metric = val_scores.get("AUROC", 0)

        print(f"Epoch {epoch:02d} | Loss={train_loss:.4f} | AUROC={val_metric:.4f} | Scores={val_scores}")

        # Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "val_scores": val_scores,
            }, save_path)

            print(f"Saved new best model to {save_path}\n")

        else:
            patience_counter += 1
            print(f"Patience = {patience_counter}/{patience}\n")

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.pkl")
    parser.add_argument("--val", required=True, help="Path to val.pkl")
    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_rbps", type=int, default=1)

    args = parser.parse_args()

    train(
        train_file=args.train,
        val_file=args.val,
        save_path=args.save,
        num_rbps=args.num_rbps,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        lr=args.lr,
    )
