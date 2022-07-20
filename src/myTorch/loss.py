#!/usr/bin/env python3
import torch
from torch import nn

def EventWeightedBinaryCrossEntropyWithLogitsLoss(input, target, weight=None):
    p = torch.ones_like(input)
    sig_x = torch.sigmoid(input)
    log_sig_x = torch.log(sig_x)
    sub_1_x = torch.sub(p, sig_x)
    sub_1_y = torch.sub(p, target)
    log_1_x = torch.log(sub_1_x)
    output = torch.neg(
        torch.add(
            torch.mul(target, log_sig_x),
            torch.mul(sub_1_y, log_1_x),
        ),
    )

    if weight != None:
        output = torch.mul(weight, output)

    return torch.div(output.sum(), weight.sum())


def EventWeightedBinaryCrossEntropy(input, target, weight=None):
    p = torch.ones_like(input)
    log_sig_x = torch.log(input)
    sub_1_x = torch.sub(p, input)
    sub_1_y = torch.sub(p, target)
    log_1_x = torch.log(sub_1_x)
    output = torch.neg(
        torch.add(
            torch.mul(target, log_sig_x),
            torch.mul(sub_1_y, log_1_x),
        ),
    )

    if weight != None:
        output = torch.mul(weight, output)

    return torch.div(output.sum(), weight.sum())


class EventWeightedBCEWithLogitsLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self):
        super(EventWeightedBCEWithLogitsLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return EventWeightedBinaryCrossEntropyWithLogitsLoss(input, target, weight)


class EventWeightedBCE(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self):
        super(EventWeightedBCE, self).__init__()

    def forward(self, input, target, weight=None):
        return EventWeightedBinaryCrossEntropy(input, target, weight)