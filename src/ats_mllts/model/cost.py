from __future__ import unicode_literals, print_function, division
import torch
import torch.nn.functional as F
from model.datautil import fill_ignore_value

debugging=False

def tag_cost(logits, tag_targets, x_mask=None, ignore_index=-100):
    '''
        logits: logits
        Assumption: batch dim comes first
    '''
    # Loss
    proba = logits.reshape(-1, logits.shape[-1]) # (N, C) shape
    if x_mask:
        tag_targets = fill_ignore_value(tag_targets, x_mask, ignore_value=ignore_index)
    tag_targets_flat = tag_targets.reshape(-1)
    if debugging:
        if not torch.any(torch.logical_or(tag_targets_flat >= 0, \
                                        tag_targets_flat == ignore_index)):
            assert "Target class index underflow."
        if torch.any(tag_targets_flat >= proba.shape[-1]):
            assert "Target class index overflow."
    loss = F.cross_entropy(proba, tag_targets_flat.long(), ignore_index=ignore_index)
    return loss

def tag_acc(tag_proba, tag_targets, x_mask):
    '''
        tag_proba: either proba or logits
    '''
    # Acc
    tag_hat = tag_proba.argmax(dim=-1, keepdim=False)
    acc = torch.sum(torch.eq(tag_hat, tag_targets)*x_mask, dim=-1, keepdim=True)
    total = torch.sum(x_mask, dim=-1, keepdim=True) # Over length dim
    # Average over length dim and batch dim
    acc = torch.sum(acc.data.float() / total.data.float()) / tag_targets.shape[0]
    return acc
