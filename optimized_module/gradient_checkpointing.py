import torch

class Checkpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fn, x):
        ctx.save_for_backward(x)
        ctx.fn = fn
        ctx.device = x.device
        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.fwd_cuda_state = torch.cuda.get_rng_state(ctx.device)
        with torch.no_grad():
            out = fn(x)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        x = ctx.saved_tensors[0].detach().requires(True)
        with torch.random.fork_rng([ctx.device]), torch.enable_grad():
            torch.set_rng_state(ctx.fwd_cpu_state)
            torch.cuda.set_rng_state(ctx.fwd_cuda_state, ctx.device)
            torch.autograd.backward(ctx.fn(x), dout)
        return None, x.grad
