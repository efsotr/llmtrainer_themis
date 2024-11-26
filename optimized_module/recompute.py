import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, Union

import torch
from torch import Tensor

class Packed:
    def __init__(self, fn: Callable[[], Tensor], stride):
        self.fn = fn
        self.stride = stride

    def unpack(self):
        return self.fn().as_strided(*self.stride).contiguous()

USE_RECOMPUTE = True

def set_recompute_flag(flag: bool):
    global USE_RECOMPUTE
    USE_RECOMPUTE = flag

@contextmanager
def recompute_flag(flag: bool):
    global USE_RECOMPUTE
    prev_flag = USE_RECOMPUTE
    USE_RECOMPUTE = flag
    try:
        yield
    finally:
        USE_RECOMPUTE = prev_flag


class Frame():
    recompute_fns : Dict[int, Any] = {}

    def add(self, x : Tensor, fn):
        if USE_RECOMPUTE is False:
            return
        self.recompute_fns[x.data_ptr()] = fn
        weakref.finalize(x.untyped_storage(), self.remove, x.data_ptr())

    def clear(self):
        self.recompute_fns.clear()

    def remove(self, ptr: int):
        self.recompute_fns.pop(ptr)
    
    @torch.no_grad
    def pack(self, x : Tensor):
        ptr = x.data_ptr() - x.storage_offset() * x.element_size()
        if ptr in self.recompute_fns:
            return Packed(self.recompute_fns[ptr], (x.size(), x.stride(), x.storage_offset()))
        return x
    
    @torch.no_grad
    def unpack(self, x: Union[Packed, Tensor]):
        if isinstance(x, Packed):
            return x.unpack()
        return x
    
frame = Frame()
torch._C._autograd._push_saved_tensors_default_hooks(
    frame.pack, frame.unpack
)

class OnceCallFn():
    def __init__(self, fn, choice_fn, /, *args, **kwargs):
        args = [frame.pack(arg) for arg in args]
        kwargs = {k: frame.pack(arg) for k, arg in kwargs.items()}

        self.fn = fn
        self.choice_fn = choice_fn
        self.args = {"args": args, "kwargs": kwargs}
        self.result = None
    
    def __call__(self):
        if self.result is not None:
            return self.result
        args = [frame.unpack(arg) for arg in self.args.pop("args")]
        kwargs = {k: frame.unpack(arg) for k, arg in self.args.pop("kwargs").items()}
        self.result = self.choice_fn(self.fn(*args, **kwargs))
        return self.result