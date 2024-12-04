import torch
from torch import nn
from functools import partial

class EffIndexSelect(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, dim: int, index: torch.Tensor):        
        output = input.index_select(dim, index)

        if torch.enable_grad():
            from .recompute import frame, OnceCallFn
            frame.add(output, OnceCallFn(partial(torch.index_select, dim=dim, index=index), lambda x: x, input.detach()))

        return output

def EffIndexSelectFn(input: torch.Tensor, dim: int, index: torch.Tensor):
    output = input.index_select(dim, index)

    if torch.enable_grad():
        from .recompute import frame, OnceCallFn
        frame.add(output, OnceCallFn(partial(torch.index_select, dim=dim, index=index), lambda x: x, input.detach()))

    return output