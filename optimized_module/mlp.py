import torch
from torch import nn

from .activation import ACT2FN

class EffActMLP(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        if config is None:
            return
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]()

    @classmethod
    def from_ori(cls, m: nn.Module):
        instance = cls(None)
        instance.config = m.config
        instance.hidden_size = m.hidden_size
        instance.intermediate_size = m.intermediate_size
        instance.gate_proj = m.gate_proj
        instance.up_proj = m.up_proj
        instance.down_proj = m.down_proj
        instance.act_fn = ACT2FN[m.config.hidden_act]()
        return instance
    
    def forward(self, x):
        out = self.down_proj(self.act_fn(self.gate_proj(x), self.up_proj(x)))
        return out
