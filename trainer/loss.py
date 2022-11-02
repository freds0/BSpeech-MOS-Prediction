import torch
import torch.nn.functional as F

def mse_loss(output, target):
    return F.mse_loss(output.squeeze(), target)

"""
def pearson_correlation_loss(output, target):
    '''
    Source-code: https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/4
    '''
    vx = output - torch.mean(output)
    vy = target - torch.mean(target)
    return torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))  # use Pearson correlation

def spearman_correlation_loss(output, target, regularization="l2", regularization_strength=1.0):
    '''
    Source-code: https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287
    '''
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])

def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

"""