import torch.nn.functional as F
import torch

def mse_loss(output, target):
    return F.mse_loss(output.squeeze(), target)

def weighted_mse_loss(inputs, targets, weights=None):
    inputs = inputs.squeeze()
    weights = weights.squeeze()
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    inputs = inputs.squeeze()
    weights = weights.squeeze()
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    inputs = inputs.squeeze()
    weights = weights.squeeze()
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    inputs = inputs.squeeze()
    weights = weights.squeeze()
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    inputs = inputs.squeeze()
    weights = weights.squeeze()
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return 


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
