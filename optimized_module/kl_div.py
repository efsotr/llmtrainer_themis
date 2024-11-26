import torch
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(log_p_ptr, log_q_ptr, output_ptr,
                 N, M,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    log_p_ptr += pid * M
    log_q_ptr += pid * M
    offset = tl.arange(0, BLOCK_SIZE)
    
    acc = 0.

    for j in range(0, M, BLOCK_SIZE):
        log_p = tl.load(log_p_ptr + j + offset, mask=offset < M - j).to(tl.float32)
        log_q = tl.load(log_q_ptr + j + offset, mask=offset < M - j).to(tl.float32)
        acc += tl.sum(tl.exp(log_p) * (log_p - log_q))

    tl.store(output_ptr + pid, acc)

def kl_div_triton(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    assert log_p.shape == log_q.shape, "log_p and log_q must have the same shape"
    N, M = log_p.shape
    output = torch.empty(N, device=log_p.device, dtype=log_p.dtype)

    BLOCK_SIZE = 1024 

    grid = (N,)

    kl_div_kernel[grid](
        log_p, log_q, output,
        N, M,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32
    )

    return output

def kl_div(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,ij->i", log_p.exp(), log_p - log_q)
