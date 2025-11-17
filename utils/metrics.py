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
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_hat)
    
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
    if np.isnan(rho):
        rho = 0.0

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        "Spearman": rho,
        "MAE": mae
    }
