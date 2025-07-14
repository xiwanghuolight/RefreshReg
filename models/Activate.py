import torch
import torch.nn as nn
import torch.nn.functional as F

class Activate(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx):
        ctx = ctx * (torch.tanh(F.softplus(ctx)))
        return ctx

    @staticmethod
    def backward(ctx, grad_output):
        input_grad = (torch.exp(ctx) * (4 * (ctx + 1) + 4 * torch.exp(2 * ctx) + torch.exp(3 * ctx) +
                                        torch.exp(ctx) * (4 * ctx + 6))) / (2 * torch.exp(ctx) + torch.exp(2 * ctx) + 2)
        return input_grad