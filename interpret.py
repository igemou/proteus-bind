import numpy as np
import torch
from torch.autograd import grad

import os
import argparse
from tqdm.auto import tqdm

from models.model import ProteusModel
from datasets.rbp_dataset import RBPDataset



ALPHABET = ["A", "C", "G", "U"]

def one_hot_to_seq(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    idx = arr.argmax(axis=-1)
    return "".join(ALPHABET[i] for i in idx)


def aggregate_nucleotide_importance(attr):
    # attr: (1, L, 4) or (L, 4)
    if attr.dim() == 3:
        attr = attr.squeeze(0)
    return attr.abs().sum(dim=-1)  # (L,)


def saliency_map(model, seq, rbns_seq, rbp_id, target):
    """
    seq:      (1, L, 4) one-hot eCLIP sequence
    rbns_seq: (1, L, 4) one-hot RBNS sequence
    rbp_id:   (1,)
    target:   "binding" or "functional"
    """
    seq = seq.clone().detach().requires_grad_(True)
    rbns_seq = rbns_seq.clone().detach().requires_grad_(True)
    rbp_id = rbp_id.clone().detach()

    bind_pred, func_pred, _ = model(seq, motif=rbns_seq, rbp_id=rbp_id)
    out = bind_pred.sum() if target == "binding" else func_pred.sum()

    model.zero_grad(set_to_none=True)
    g_seq, g_rbns = grad(out, [seq, rbns_seq], retain_graph=False)

    return g_seq.squeeze(0), g_rbns.squeeze(0)


def integrated_gradients(model, seq, rbns_seq, rbp_id, target, steps=50):
    """
    Integrated Gradients wrt eCLIP and RBNS sequences.

    seq:      (1, L, 4)
    rbns_seq: (1, L, 4)
    """
    device = seq.device

    baseline_seq = torch.zeros_like(seq)
    baseline_rbns = torch.zeros_like(rbns_seq)

    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    grads_seq = []
    grads_rbns = []

    for a in alphas:
        x = (baseline_seq + a * (seq - baseline_seq)).detach().requires_grad_(True)
        m = (baseline_rbns + a * (rbns_seq - baseline_rbns)).detach().requires_grad_(True)

        bind_pred, func_pred, _ = model(x, motif=m, rbp_id=rbp_id)
        out = bind_pred.sum() if target == "binding" else func_pred.sum()

        model.zero_grad(set_to_none=True)
        g_seq, g_rbns = grad(out, [x, m])
        grads_seq.append(g_seq)
        grads_rbns.append(g_rbns)

    avg_seq = torch.stack(grads_seq).mean(dim=0)
    avg_rbns = torch.stack(grads_rbns).mean(dim=0)

    ig_seq = (seq - baseline_seq) * avg_seq
    ig_rbns = (rbns_seq - baseline_rbns) * avg_rbns

    return ig_seq.squeeze(0), ig_rbns.squeeze(0)


def in_silico_mutagenesis(model, seq, rbns_seq, rbp_id, target):
    """
    Returns:
      delta_max:  (L,)
      delta_mean: (L,)
      full_matrix: (L, 4)
    """
    device = seq.device
    seq = seq.clone().detach()

    with torch.no_grad():
        bind_pred, func_pred, _ = model(seq, motif=rbns_seq, rbp_id=rbp_id)
        base = bind_pred.item() if target == "binding" else func_pred.item()

    L = seq.shape[1]
    full = torch.zeros((L, 4), device=device)

    # original base indices
    orig_idx = seq[0].argmax(dim=-1).cpu().numpy()

    for i in range(L):
        for b in range(4):
            if b == orig_idx[i]:
                continue
            mut = seq.clone()
            mut[0, i, :] = 0
            mut[0, i, b] = 1

            with torch.no_grad():
                bind_pred, func_pred, _ = model(mut, motif=rbns_seq, rbp_id=rbp_id)
                new = bind_pred.item() if target == "binding" else func_pred.item()

            full[i, b] = new - base

    return full.abs().max(dim=-1).values, full.abs().mean(dim=-1), full


def rbp_embedding_importance(model, seq, rbns_seq, rbp_id, target):
    rbp_id = rbp_id.clone().detach()
    emb = model.rbp_embed(rbp_id).clone().detach().requires_grad_(True)

    h_seq = model.seq_encoder(seq)
    h_rbns = model.rbns_encoder(rbns_seq)
    h = model.fusion(h_seq, emb, h_rbns)

    out_b = model.bind_head(h)
    out_f = model.func_head(h)
    out = out_b.sum() if target == "binding" else out_f.sum()

    model.zero_grad(set_to_none=True)
    g = grad(out, emb)[0].detach()
    return g.squeeze(0)  # (rbp_emb_dim,)


def interpret_sample(model, dataset, idx, device):
    """
    Run all attribution methods for a single sample:
      - saliency (binding, functional)
      - integrated gradients (binding, functional)
      - in silico mutagenesis (binding, functional)
      - RBP embedding gradients (binding, functional)
    """
    sample = dataset[idx]
    (
        seq_oh,      # (L, 4)
        rbns_oh,     # (L, 4)
        rbns_aff,    # (motif_dim,)
        rbp_id,      # scalar LongTensor
        y_bind,      # (1,)
        y_func,      # (1,)
        y_next,      # ()
    ) = sample

    seq_oh = seq_oh.unsqueeze(0).to(device)      # (1, L, 4)
    rbns_oh = rbns_oh.unsqueeze(0).to(device)    # (1, L, 4)
    rbp_id = rbp_id.unsqueeze(0).to(device)      # (1,)

    seq_str = one_hot_to_seq(seq_oh[0])

    results = dict(seq=seq_str)

    sal_b_seq, sal_b_rbns = saliency_map(model, seq_oh, rbns_oh, rbp_id, "binding")
    sal_f_seq, sal_f_rbns = saliency_map(model, seq_oh, rbns_oh, rbp_id, "functional")

    results["sal_b_seq"] = sal_b_seq.cpu().numpy()
    results["sal_f_seq"] = sal_f_seq.cpu().numpy()
    results["sal_b_imp"] = aggregate_nucleotide_importance(sal_b_seq).cpu().numpy()
    results["sal_f_imp"] = aggregate_nucleotide_importance(sal_f_seq).cpu().numpy()

    results["sal_b_rbns"] = sal_b_rbns.cpu().numpy()
    results["sal_f_rbns"] = sal_f_rbns.cpu().numpy()

    ig_b_seq, ig_b_rbns = integrated_gradients(model, seq_oh, rbns_oh, rbp_id, "binding")
    ig_f_seq, ig_f_rbns = integrated_gradients(model, seq_oh, rbns_oh, rbp_id, "functional")

    results["ig_b_seq"] = ig_b_seq.cpu().numpy()
    results["ig_f_seq"] = ig_f_seq.cpu().numpy()
    results["ig_b_imp"] = aggregate_nucleotide_importance(ig_b_seq).cpu().numpy()
    results["ig_f_imp"] = aggregate_nucleotide_importance(ig_f_seq).cpu().numpy()

    results["ig_b_rbns"] = ig_b_rbns.cpu().numpy()
    results["ig_f_rbns"] = ig_f_rbns.cpu().numpy()

    ism_b_max, ism_b_mean, ism_b_full = in_silico_mutagenesis(model, seq_oh, rbns_oh, rbp_id, "binding")
    ism_f_max, ism_f_mean, ism_f_full = in_silico_mutagenesis(model, seq_oh, rbns_oh, rbp_id, "functional")

    results["ism_b_max"] = ism_b_max.cpu().numpy()
    results["ism_b_mean"] = ism_b_mean.cpu().numpy()
    results["ism_b_full"] = ism_b_full.cpu().numpy()

    results["ism_f_max"] = ism_f_max.cpu().numpy()
    results["ism_f_mean"] = ism_f_mean.cpu().numpy()
    results["ism_f_full"] = ism_f_full.cpu().numpy()

    emb_b = rbp_embedding_importance(model, seq_oh, rbns_oh, rbp_id, "binding")
    emb_f = rbp_embedding_importance(model, seq_oh, rbns_oh, rbp_id, "functional")

    results["emb_b"] = emb_b.cpu().numpy()
    results["emb_f"] = emb_f.cpu().numpy()

    return results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--split", required=True, help="Path to split .pkl (e.g., test split for one RBP)")
    parser.add_argument("--pos_label", required=True, help="Path to positive functional label .pkl for this RBP")
    parser.add_argument("--neg_label", required=True, help="Path to negative functional label .pkl for this RBP")
    parser.add_argument("--num_rbps", type=int, required=True, help="Number of RBPs used during training")
    parser.add_argument("--out_dir", required=True, help="Directory to save .npz attribution files")
    parser.add_argument("--num_samples", type=int, default=20, help="How many samples to interpret")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index in the dataset")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = RBPDataset(
        all_split_files=[args.split],
        all_pos_label_files=[args.pos_label],
        all_neg_label_files=[args.neg_label],
        target_rbp_id=None,
        multi_RBP=True,  
    )

    motif_dim = 64

    model = ProteusModel(
        hidden=128,
        motif_dim=motif_dim,
        num_rbps=args.num_rbps,
        rbp_emb_dim=32,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    end = min(args.start_idx + args.num_samples, len(dataset))

    for idx in tqdm(range(args.start_idx, end), desc="Interpreting"):
        res = interpret_sample(model, dataset, idx, device)
        out_path = os.path.join(args.out_dir, f"sample_{idx:05d}.npz")
        np.savez_compressed(out_path, **res)


if __name__ == "__main__":
    main()
