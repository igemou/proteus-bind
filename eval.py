import torch
from models.model import ProteusModel
from datasets.rbp_dataset import make_loader
from utils.metrics import binding_metrics, functional_metrics

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()

    all_true_b, all_pred_b = [], []
    all_true_f, all_pred_f = [], []

    for seq, rbns_vec, rbp_ids, expr_changes in loader:
        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)

        pred_b, pred_f = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        true_b = torch.ones(seq.size(0), 1, device=device)
        all_true_b.append(true_b.cpu())
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved checkpoint (.pt)")
    parser.add_argument("--test", required=True, help="Path to test.pkl")
    parser.add_argument("--num_rbps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = make_loader(args.test, batch_size=args.batch, shuffle=False)

    first_motif = test_loader.dataset.data[0][1][1]
    motif_dim = len(first_motif)

    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=args.num_rbps,
        rbp_emb_dim=32
    ).to(device)

    print(f"Loading checkpoint from {args.model}...")
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print("\nRunning evaluation...\n")
    results = evaluate(model, test_loader, device)

    print("====== FINAL TEST METRICS ======")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
