import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import spearmanr


def binding_metrics(y_true, y_pred):
    """
    AUROC, AUPRC, F1 for binary binding prediction.
    y_pred is a probability (after sigmoid).
    """
    y_true = y_true.detach().cpu().numpy().ravel()
    y_prob = y_pred.detach().cpu().numpy().ravel()
    y_hat = (y_prob > 0.5).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.0

    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    try:
        f1 = f1_score(y_true, y_hat)
    except:
        f1 = 0.0
    
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": f1
    }

def functional_metrics(y_true, y_pred):
    """
    Spearman correlation and MAE for functional regression.
    """
    y_true = y_true.detach().cpu().numpy().ravel()
    y_pred = y_pred.detach().cpu().numpy().ravel()

    # Spearman
    rho, _ = spearmanr(y_true, y_pred)
    rho = 0.0 if np.isnan(rho) else float(rho)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        "Spearman": rho,
        "MAE": mae
    }

def next_base_metrics(y_true, y_pred_logits):
    """
    CrossEntropyLoss, accuracy
    """
    y_true = y_true.detach().cpu()            # (B,)
    logits = y_pred_logits.detach().cpu()     # (B, 4)

    ce_loss = torch.nn.functional.cross_entropy(logits, y_true).item()

    pred_class = logits.argmax(dim=-1)         # (B,)
    acc = (pred_class == y_true).float().mean().item()

    return {
        "Next_CE": ce_loss,
        "Next_Acc": acc
    }
