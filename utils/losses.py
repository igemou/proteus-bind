import torch
import torch.nn as nn
import torch.nn.functional as F


bce = nn.BCELoss()

def binding_loss(pred_bind, true_bind):
    """
    pred_bind: (B, 1), passed through sigmoid
    true_bind: (B, 1), 0/1
    """
    return bce(pred_bind, true_bind.float())

mse = nn.MSELoss()

def functional_loss(pred_func, true_func, mask):
    """
    pred_func: (B,1)
    true_func: (B,1)
    mask:      (B,) — 1 means valid ΔΨ, 0 means missing
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_func.device)

    pred = pred_func[mask == 1]
    true = true_func[mask == 1]

    return mse(pred, true.float())


def multitask_loss(pred_bind, true_bind, pred_func, true_func, mask, lambda_bind=1.0, lambda_func=0.5):
    """     
    L = λ_bind * BCE + λ_func * MSE
    """
    L_bind = binding_loss(pred_bind, true_bind)
    L_func = functional_loss(pred_func, true_func, mask)

    return (lambda_bind * L_bind) + (lambda_func * L_func), L_bind, L_func
