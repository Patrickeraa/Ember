import torch
import torch.nn as nn


def compute_grad_stats(model):
    total_norm = 0.0
    min_grad, max_grad = float('inf'), float('-inf')
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.data
        param_norm = grad.norm(2).item()
        total_norm += param_norm ** 2
        min_grad = min(min_grad, grad.min().item())
        max_grad = max(max_grad, grad.max().item())
    total_norm = total_norm ** 0.5
    return total_norm, min_grad, max_grad

def compute_weight_stats(model):
    total_norm = 0.0
    min_w, max_w = float('inf'), float('-inf')
    for p in model.parameters():
        w = p.data
        param_norm = w.norm(2).item()
        total_norm += param_norm ** 2
        min_w = min(min_w, w.min().item())
        max_w = max(max_w, w.max().item())
    total_norm = total_norm ** 0.5
    return total_norm, min_w, max_w