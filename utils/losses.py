import torch
import torch.nn as nn

bce = nn.BCELoss()
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

def binding_loss(pred, true):
    return bce(pred, true)

def functional_loss(pred, true):
    return mse(pred, true)

def nextbase_loss(logits, true_next):
    """
    logits: (B, 4) raw scores from NextBaseHead
    true_next: (B,) integer targets in {0,1,2,3}
    """
    return ce(logits, true_next)

def multitask_loss(pred_b, true_b, pred_f, true_f, pred_next, true_next,
                    lambda_bind=1.0, lambda_func=0.5, lambda_next=0.5):

    Lb = binding_loss(pred_b, true_b)

    if lambda_func > 0.0 and pred_f is not None and true_f is not None:
        Lf = functional_loss(pred_f, true_f)
    else:
        Lf = torch.tensor(0.0, device=pred_b.device)

    if lambda_next > 0.0 and pred_next is not None and true_next is not None:
        Ln = nextbase_loss(pred_next, true_next)
    else:
        Ln = torch.tensor(0.0, device=pred_b.device)

    total = lambda_bind * Lb + lambda_func * Lf + lambda_next * Ln
    return total, Lb, Lf, Ln
