import torch
from torch import nn
from functools import partial

def to_dtype(input, dtype):
    return input.to(dtype)

class EffToDtype(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, dtype: torch.dtype):
        if input.dtype == dtype:
            return input
        
        output = input.to(dtype)

        if torch.enable_grad():
            from .recompute import frame, OnceCallFn
            frame.add(output, OnceCallFn(partial(to_dtype, dtype), lambda x: x, input.detach()))

        return output

def EffToDtypeFn(input: torch.Tensor, dtype: torch.dtype):
    if input.dtype == dtype:
            return input
    
    output = input.to(dtype)

    if torch.enable_grad():
        from .recompute import frame, OnceCallFn
        frame.add(output, OnceCallFn(partial(to_dtype, dtype), lambda x: x, input.detach()))

    return output