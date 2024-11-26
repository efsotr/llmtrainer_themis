import os
import pytest
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from activation import swiglu_func

has_cuda = "CUDA_VISIBLE_DEVICES" not in os.environ
device = "cuda" if has_cuda else "cpu"

def copy(x: Tensor):
    return x.detach().clone().requires_grad_(True)

def swiglu_act_ref(x: Tensor, y: Tensor):
    return (F.silu(x.float()) * y.float()).to(x.dtype)

def swiglu_act(x: Tensor, y: Tensor):
    return F.silu(x) * y

def forward(func, dout, args):
    out = func(*args)
    if dout is None:
        dout = torch.randn_like(out)
    out.backward(dout)
    return out, [arg.grad for arg in args], dout

fast = swiglu_func()

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [64 * i for i in range(1, 33)])
def test_gelu_act(dtype, seqlen):
    x = torch.randn((seqlen, ), dtype=dtype, device=device, requires_grad=True)
    y = torch.randn((seqlen, ), dtype=dtype, device=device, requires_grad=True)

    args = (x, y)
    out_ref, grad_ref, dout = forward(swiglu_act_ref, None, args)
    out_hf, grad_hf, dout = forward(swiglu_act, dout, args)
    out_fast, grad_fast, dout = forward(fast, dout, args)

    assert (out_fast.float() - out_ref.float()).abs().max().item() <= (out_hf.float() - out_ref.float()).abs().max().item() * 2
    for i in range(len(grad_ref)):
        assert (grad_fast[i].float() - grad_ref[i].float()).abs().max().item() <= (grad_hf[i].float() - grad_ref[i].float()).abs().max().item() * 2