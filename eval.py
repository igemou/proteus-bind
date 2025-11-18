import torch
from models.model import ProteusModel
from datasets.rbp_dataset import make_loader
from utils.metrics import binding_metrics, functional_metrics


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()

    all_true_b, all_pred_b = [], []
    all_true_f, all_pred_f = [], []

    for seq, rbns_vec, rbp_ids, func_label, bind_label in loader:

        seq = seq.to(device)
        rbns_vec = rbns_vec.to(device)
        rbp_ids = rbp_ids.to(device)
        bind_label = bind_label.to(device)
        func_label = func_label.to(device)

        pred_b, pred_f = model(seq, motif=rbns_vec, rbp_id=rbp_ids)

        # collect binding
        all_true_b.append(bind_label.cpu())
        all_pred_b.append(pred_b.cpu())

        # collect functional
        all_true_f.append(func_label.cpu())
        all_pred_f.append(pred_f.cpu())

    results = {}

    # Binding metrics
    yb = torch.cat(all_true_b)
    pb = torch.cat(all_pred_b)
    results.update(binding_metrics(yb, pb))

    # Functional metrics
    yf = torch.cat(all_true_f)
    pf = torch.cat(all_pred_f)
    results.update(functional_metrics(yf, pf))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved checkpoint (.pt)")
    parser.add_argument("--test", required=True, help="Path to test split (.pkl)")
    parser.add_argument("--pos_label", required=True, help="Positive functional label (.pkl)")
    parser.add_argument("--neg_label", required=True, help="Negative functional label (.pkl)")
    parser.add_argument("--num_rbps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = make_loader(
        split=args.test,
        pos=args.pos_label,
        neg=args.neg_label,
        batch_size=args.batch,
        shuffle=False
    )

    example_aff = test_loader.dataset.data[0][1][1]   # (rbns_seq, affinity_vec)
    motif_dim = len(example_aff)

    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=args.num_rbps,
        rbp_emb_dim=32
    ).to(device)

    print(f"Loading checkpoint from {args.model}...")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])

    print("\nRunning evaluation...\n")
    results = evaluate(model, test_loader, device)

    print("====== FINAL TEST METRICS ======")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
