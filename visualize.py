import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

ALPHABET = ["A", "C", "G", "U"]
sns.set(style="whitegrid", font_scale=1.3)


def plot_ig_heatmap(ig_seq, seq, save_path, title):
    ig_seq = np.squeeze(ig_seq)  # (L, 4)
    if ig_seq.ndim != 2:
        raise ValueError(f"ig_seq should be (L,4), got shape {ig_seq.shape}")

    plt.figure(figsize=(14, 3))
    sns.heatmap(
        ig_seq.T,
        cmap="viridis",
        xticklabels=list(seq),
        yticklabels=ALPHABET,
        cbar_kws={"label": "IG attribution"},
    )
    plt.title(title)
    plt.xlabel("Sequence position")
    plt.ylabel("Nucleotide")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_position_importance(importance, seq, save_path, title):
    importance = np.squeeze(importance)
    plt.figure(figsize=(14, 3))
    plt.bar(range(len(seq)), importance, color="darkblue")
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ism_heatmap(ism_matrix, seq, save_path, title):
    ism_matrix = np.squeeze(ism_matrix)  # (L,4)
    if ism_matrix.ndim != 2:
        raise ValueError(f"ism_matrix should be (L,4), got shape {ism_matrix.shape}")

    plt.figure(figsize=(14, 3))
    sns.heatmap(
        ism_matrix.T,
        cmap="coolwarm",
        center=0.0,
        xticklabels=list(seq),
        yticklabels=ALPHABET,
        cbar_kws={"label": "Δ prediction"},
    )
    plt.title(title)
    plt.xlabel("Sequence position")
    plt.ylabel("Mutated nucleotide")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_motif_contribution(motif_attr, save_path, title):
    motif_attr = np.array(motif_attr)
    motif_attr = np.squeeze(motif_attr)

    if motif_attr.ndim == 2:
        motif_attr = np.abs(motif_attr).sum(axis=-1)
    elif motif_attr.ndim > 2:
        raise ValueError(f"Unexpected motif_attr shape {motif_attr.shape}")

    plt.figure(figsize=(10, 3))
    plt.bar(range(len(motif_attr)), motif_attr, color="purple")
    plt.title(title)
    plt.xlabel("RBNS position")
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rbp_embedding(emb_attr, save_path, title):
    emb_attr = np.squeeze(emb_attr)
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(emb_attr)), emb_attr, color="darkgreen")
    plt.title(title)
    plt.xlabel("Embedding dimension")
    plt.ylabel("Gradient magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_dir",
        required=True,
        help="Directory containing interpret outputs (sample_XXXXX.npz)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Where to save aggregated plots",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files found in {args.npz_dir}")

    length_counts = {}
    lengths_per_file = {}

    for fname in npz_files:
        path = os.path.join(args.npz_dir, fname)
        data = np.load(path, allow_pickle=True)
        seq = data["seq"].item() if isinstance(data["seq"], np.ndarray) else str(data["seq"])
        seq = str(seq)
        L = len(seq)
        lengths_per_file[fname] = L
        length_counts[L] = length_counts.get(L, 0) + 1

    target_len = max(length_counts.items(), key=lambda kv: kv[1])[0]
    print("Sequence length distribution:", length_counts)
    print(f"Using target length {target_len} (N={length_counts[target_len]} samples)")

    agg = {}
    seq_ref = None
    n = 0

    keys = [
        "ig_b_seq",
        "ig_f_seq",
        "ig_b_imp",
        "ig_f_imp",
        "ism_b_full",
        "ism_f_full",
        "ig_b_rbns",
        "ig_f_rbns",
        "emb_b",
        "emb_f",
    ]

    for fname in npz_files:
        if lengths_per_file[fname] != target_len:
            continue

        path = os.path.join(args.npz_dir, fname)
        data = np.load(path, allow_pickle=True)

        seq = data["seq"].item() if isinstance(data["seq"], np.ndarray) else str(data["seq"])
        seq = str(seq)

        if seq_ref is None:
            seq_ref = seq

        tmp = {}
        if not agg:
            for k in keys:
                tmp[k] = np.array(data[k], dtype=np.float64)
        else:
            skip_this = False
            for k in keys:
                arr = np.array(data[k], dtype=np.float64)
                if agg[k].shape != arr.shape:
                    print(
                        f"Skipping {fname} for key '{k}': "
                        f"shape {arr.shape} != agg {agg[k].shape}"
                    )
                    skip_this = True
                    break
                tmp[k] = arr
            if skip_this:
                continue

        if not agg:
            agg = {k: v.copy() for k, v in tmp.items()}
        else:
            for k in keys:
                agg[k] += tmp[k]

        n += 1

    if n == 0:
        raise SystemExit(
            f"No samples with consistent shapes for length {target_len}; nothing to aggregate."
        )

    for k in agg:
        agg[k] /= float(n)

    print(f"Averaged over {n} samples with length {target_len}.")

    seq = seq_ref

    # IG heatmaps
    plot_ig_heatmap(
        agg["ig_b_seq"],
        seq,
        os.path.join(args.outdir, "ig_binding_heatmap_avg.png"),
        "Integrated Gradients — Binding Task (average)",
    )
    plot_ig_heatmap(
        agg["ig_f_seq"],
        seq,
        os.path.join(args.outdir, "ig_functional_heatmap_avg.png"),
        "Integrated Gradients — Functional Task (average)",
    )

    # position-wise IG importance
    plot_position_importance(
        agg["ig_b_imp"],
        seq,
        os.path.join(args.outdir, "ig_binding_importance_avg.png"),
        "Binding Task — IG Position-wise Importance (average)",
    )
    plot_position_importance(
        agg["ig_f_imp"],
        seq,
        os.path.join(args.outdir, "ig_functional_importance_avg.png"),
        "Functional Task — IG Position-wise Importance (average)",
    )

    # ISM matrices
    plot_ism_heatmap(
        agg["ism_b_full"],
        seq,
        os.path.join(args.outdir, "ism_binding_heatmap_avg.png"),
        "In Silico Mutagenesis — Binding Task (average)",
    )
    plot_ism_heatmap(
        agg["ism_f_full"],
        seq,
        os.path.join(args.outdir, "ism_functional_heatmap_avg.png"),
        "In Silico Mutagenesis — Functional Task (average)",
    )

    # RBNS sequence attribution
    plot_motif_contribution(
        agg["ig_b_rbns"],
        os.path.join(args.outdir, "motif_binding_attribution_avg.png"),
        "RBNS Sequence Attribution — Binding Task (average)",
    )
    plot_motif_contribution(
        agg["ig_f_rbns"],
        os.path.join(args.outdir, "motif_functional_attribution_avg.png"),
        "RBNS Sequence Attribution — Functional Task (average)",
    )

    # RBP embedding attribution
    plot_rbp_embedding(
        agg["emb_b"],
        os.path.join(args.outdir, "rbp_embedding_binding_avg.png"),
        "RBP Embedding Attribution — Binding Task (average)",
    )
    plot_rbp_embedding(
        agg["emb_f"],
        os.path.join(args.outdir, "rbp_embedding_functional_avg.png"),
        "RBP Embedding Attribution — Functional Task (average)",
    )

if __name__ == "__main__":
    main()
