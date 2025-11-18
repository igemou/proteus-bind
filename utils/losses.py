import torch
import torch.nn as nn

bce = nn.BCELoss()
mse = nn.MSELoss()

def binding_loss(pred, true):
    return bce(pred, true)

def functional_loss(pred, true):
    return mse(pred, true)

def multitask_loss(pred_b, true_b, pred_f, true_f,
                   lambda_bind=1.0, lambda_func=0.5):

    Lb = binding_loss(pred_b, true_b)
    Lf = functional_loss(pred_f, true_f)

    total = lambda_bind * Lb + lambda_func * Lf
    return total, Lb, Lf
