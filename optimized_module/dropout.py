import torch
from torch import nn, Tensor
import triton
import triton.language as tl
from functools import partial

@triton.jit
def dropout_forward_kernel(
    x_ptr,        # Pointer to input tensor x
    y_ptr,        # Pointer to output tensor y
    p,            # Dropout probability
    scale,        # 
    seed,         # Random seed
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size for Triton kernel
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    rand = tl.rand(seed, offsets)
    x = tl.where(rand > p, x, 0.0) * scale
    tl.store(y_ptr + offsets, x, mask=mask)

class TritonDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed=None):
        n_elements = x.numel()
        y = torch.empty_like(x)

        if seed is None:
            seed = torch.randint(0, 2**31, (1,), dtype=torch.int64, device=x.device).item()

        # Define grid size for Triton kernel
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Launch Triton kernel for forward pass
        dropout_forward_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            p=p,
            scale=1.0 / (1.0 - p),
            seed=seed,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.seed = seed
        ctx.p = p
        return y

    @staticmethod
    def backward(ctx, dy):
        n_elements = dy.numel()
        dx = torch.empty_like(dy)

        # Define grid size for Triton kernel
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Launch Triton kernel for backward pass
        dropout_forward_kernel[grid](
            x_ptr=dy,
            y_ptr=dx,
            p=ctx.p,
            scale=1.0 / (1.0 - ctx.p),
            seed=ctx.seed,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # No gradients w.r.t p and seed
        return dx, None, None
    
def dropout_w_rng(input: Tensor, p: float, rng_state):
    with torch.random.fork_rng(devices=[input.device]):
        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state(rng_state[1], input.device)
        return TritonDropoutFunction.apply(input, p)

class EffDropout(nn.Dropout):

    @classmethod
    def from_ori(cls, m: nn.Dropout):
        assert m.inplace == False
        instance = cls(m.p, m.inplace)
        return instance

    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        
        if torch.enable_grad():
            rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state(input.device))

        output = TritonDropoutFunction.apply(input, self.p, self.training)
        
        if torch.enable_grad():
            from .recompute import frame, OnceCallFn
            frame.add(output, OnceCallFn(partial(dropout_w_rng, p=self.p, rng_state=rng_state), lambda x: x, input.detach()))

        return output