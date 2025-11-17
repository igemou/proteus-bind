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

def functional_loss(pred_func, true_func):
    """
    pred_func: (B, 1)
    true_func: (B, 1), ΔΨ or log2FC values
    """
    return mse(pred_func, true_func.float())


def multitask_loss(pred_bind, true_bind,
                   pred_func, true_func,
                   lambda_bind=1.0,
                   lambda_func=0.5):
    """
    Compute:
    L = λ_bind * BCE + λ_func * MSE
    """

    L_bind = binding_loss(pred_bind, true_bind)
    L_func = functional_loss(pred_func, true_func)

    return (lambda_bind * L_bind) + (lambda_func * L_func), L_bind, L_func
