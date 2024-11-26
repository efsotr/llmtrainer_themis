import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=32),
    ],
    key=["M"]
)
@triton.jit
def logsumexp_triton_kernel(X, Y, M : tl.constexpr, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + pid * M + offsets, mask=offsets < M, other=-8192)
    max_x = tl.max(x).to(tl.float32)
    exp_x = tl.exp(x - max_x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    logsumexp = (tl.log(sum_exp_x) + max_x).to(x.dtype)
    tl.store(Y + pid, logsumexp)

def logsumexp_triton(x: torch.Tensor, dim = -1, keepdim = False):
    assert dim == -1 and len(x.shape) >= 2
    x_ori_shape = x.shape[:-1]
    x = x.view((-1, x.size(-1)))
    N, M = x.shape
    BLOCK_SIZE = triton.next_power_of_2(M)
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    logsumexp_triton_kernel[(N, )](x, out, M, BLOCK_SIZE)

    return out.view(x_ori_shape + (1,)) if keepdim else out.view(x_ori_shape)

class Efflogsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        l = logsumexp_triton(x, dim=-1, keepdim=True)
        ctx.save_for_backward(x, l)
        return l

    @staticmethod
    def backward(ctx, dout):
        x, l = ctx.saved_tensors
        return (x - l).exp_().mul_(dout)
    
# class Efflogsoftmax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         s = x - logsumexp_triton(x, dim=-1, keepdim=True)
#         ctx.save_for_backward(s)
#         return s

#     @staticmethod
#     def backward(ctx, dout):
#         s = ctx.saved_tensors
#         return dout - s.exp().mul_(dout.sum(-1, keepdim=True))
    
# class Effsoftmax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         l = logsumexp_triton(x, dim=-1, keepdim=True)
#         p = (x - l).exp_()
#         ctx.save_for_backward(p)
#         return p

#     @staticmethod
#     def backward(ctx, dout):
#         p = ctx.saved_tensors
#         return dout - p * ((dout * p).sum(-1, keepdim=True))

@torch.jit.script
def crossentropy_fwd(x, l, idx, ignore_idx):
    pad_mask = idx == ignore_idx
    idx = idx.clamp(0)
    s = torch.gather(x, -1, idx)
    out = (l - s).masked_fill_(pad_mask, 0)
    return out.squeeze(-1), idx, pad_mask

@torch.jit.script
def crossentropy_bwd(x, l, dout, idx):
    return (x - l).exp_().mul_(dout).scatter_add_(-1, idx, -dout)

@torch.jit.script
def crossentropy_bwd_x_no_need(x, l, dout, idx):
    return x.data.sub_(l).exp_().mul_(dout).scatter_add_(-1, idx, -dout)

class EffCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, ignore_idx = -100, x_no_need_in_bwd = True):
        l = logsumexp_triton(x, dim=-1, keepdim=True)
        out, idx, pad_mask = crossentropy_fwd(x, l, idx, torch.tensor(ignore_idx, dtype=idx.dtype))
        ctx.save_for_backward(x, l, idx, pad_mask)
        ctx.x_no_need_in_bwd = x_no_need_in_bwd
        return out

    @staticmethod
    def backward(ctx, dout):
        x, l, idx, pad_mask = ctx.saved_tensors
        dout = dout.unsqueeze(-1).masked_fill(pad_mask, 0)
        dx = crossentropy_bwd_x_no_need(x, l, dout, idx) if ctx.x_no_need_in_bwd else crossentropy_bwd(x, l, dout, idx)
        return dx, None, None    