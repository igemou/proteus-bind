import os
import torch
from torch.utils.data import DataLoader

from models.model import ProteusModel
from datasets.rbp_dataset import make_loader
from utils.losses import multitask_loss
from utils.metrics import binding_metrics, functional_metrics
from tqdm.auto import tqdm

def train_one_epoch(model, loader, optim, device,
                    lambda_bind=1.0, lambda_func=0.5):

    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training")

    for seq, rbns_vec, rbp_ids, func_label, bind_label in pbar:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        bind_label = bind_label.to(device)
        func_label = func_label.to(device)

        # always single RBP for now
        rbp_ids = torch.zeros(seq.size(0), dtype=torch.long, device=device)

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

    for seq, rbns_vec, rbp_ids, func_label, bind_label in loader:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)
        func_label = func_label.to(device)
        bind_label = bind_label.to(device)

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


def train(train_file, val_file,
          pos_label_file, neg_label_file,
          save_path="best_model.pt",
          num_rbps=1, epochs=30,
          batch_size=16, patience=7,
          lr=1e-4,
          lambda_bind=1.0, lambda_func=0.5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(train_file, pos_label_file, neg_label_file,
                               batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_file, pos_label_file, neg_label_file,
                             batch_size=batch_size, shuffle=False)

    # motif dimension
    example_aff = train_loader.dataset.data[0][1][1]
    motif_dim = len(example_aff)

    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=num_rbps,
        rbp_emb_dim=32
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = -float("inf")
    patience_counter = 0

    print("\n=== Starting training ===\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optim, device,
            lambda_bind=lambda_bind,
            lambda_func=lambda_func
        )

        val_scores = evaluate(model, val_loader, device)
        val_metric = val_scores.get("AUROC", 0.0)

        print(f"Epoch {epoch:02d} | Loss={train_loss:.4f} | "
              f"AUROC={val_metric:.4f} | Scores={val_scores}")

        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "val_scores": val_scores,
            }, save_path)

            print(f"Saved best model to {save_path}\n")

        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{patience}\n")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("Training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--pos_label", required=True)
    parser.add_argument("--neg_label", required=True)
    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_rbps", type=int, default=1)

    args = parser.parse_args()

    train(
        train_file=args.train,
        val_file=args.val,
        pos_label_file=args.pos_label,
        neg_label_file=args.neg_label,
        save_path=args.save,
        num_rbps=args.num_rbps,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        lr=args.lr,
    )
