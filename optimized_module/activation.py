import torch
from torch import nn
import torch.nn.functional as F


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def gelu_fwd(x, y):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=x.dtype) * y


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def gelu_bwd(g, x, y):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return (ff * g * y).to(dtype=x.dtype), (g * (x * 0.5 * (1.0 + tanh_out))).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return gelu_fwd(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, x, y)
        return tmp
    

swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) / (1.0f + ::exp(-float(x))) * float(y);
}
"""
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)

class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)

class swiglu_func(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return SwiGLUFunction.apply(x, y)
    
class gelu_func(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return FastGeLUFunction.apply(x, y)
    

class swiglu_wrc_func(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = SwiGLUFunction.apply(x, y)
        if torch.is_grad_enabled():
            from .recompute import frame, OnceCallFn
            frame.add(out, OnceCallFn(SwiGLUFunction.apply, lambda x: x, x.detach(), y.detach()))
        return out
    
class gelu_wrc_func(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = FastGeLUFunction.apply(x, y)
        if torch.is_grad_enabled():
            from .recompute import frame, OnceCallFn
            frame.add(out, OnceCallFn(FastGeLUFunction.apply, lambda x: x, x.detach(), y.detach()))
        return out 

ACT2FN = {"silu": swiglu_wrc_func, 
          "gelu_pytorch_tanh": gelu_wrc_func}

