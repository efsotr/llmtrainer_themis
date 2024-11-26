import os
import pytest
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layer_norm import rms_norm_fn, rms_norm_ref

device = "cuda" 

def copy(x: Tensor):
    return x.detach().clone().requires_grad_(True)

def rms_ref(x, weight, bias, weight_add_one):
    return rms_norm_ref(x, weight + weight_add_one, bias, upcast=True).to(x.dtype)

def rms(x, weight, bias, weight_add_one):
    return rms_norm_ref(x, weight + weight_add_one, bias)

def forward(func, dout, kwargs):
    out = func(**kwargs)
    if dout is None:
        dout = torch.randn_like(out)
    out.backward(dout)
    return out, [arg.grad for arg in kwargs.values() if torch.is_tensor(arg)], dout

eps = 1e-6

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [2048, 3840, 4096, 8192])
@pytest.mark.parametrize("weight_add_one", [0, 1])
def test_gelu_act(dtype, seqlen, weight_add_one):
    x = torch.randn((16, seqlen), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((seqlen, ), dtype=dtype, device=device, requires_grad=True)

    args = {"x": x, "weight": w, "bias": None, "weight_add_one": weight_add_one}
    out_ref, grad_ref, dout = forward(rms_ref, None, args)
    out_hf, grad_hf, dout = forward(rms, dout, args)
    out_fast, grad_fast, dout = forward(rms_norm_fn, dout, args)

    assert (out_fast.float() - out_ref.float()).abs().max().item() <= (out_hf.float() - out_ref.float()).abs().max().item() * 2
    for i in range(len(grad_ref)):
        assert (grad_fast[i].float() - grad_ref[i].float()).abs().max().item() <= (grad_hf[i].float() - grad_ref[i].float()).abs().max().item() * 2