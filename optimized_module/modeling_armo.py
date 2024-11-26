from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.utils import ModelOutput
from transformers.utils import add_start_docstrings_to_model_forward

from .modeling_llama_multi_layers_ckpt import LlamaModel, LlamaPreTrainedModel

def to_pt(x):
    if torch.is_tensor(x):
        return x
    return torch.tensor(x)

def do_batch_conversations(conversations: List[List[Dict[str, str]]]):
    pass


def do_batch_prompts_responses(inputs: List[Tuple[List[int], List[int]]], res = None, seq_reward = True):
    if res is not None:
        inputs = list(zip(inputs, res))
    input_ids = []
    position_ids = []
    gate_mask = []
    gate_idx = []
    seqlens = []
    for idx, (prompt, response) in enumerate(inputs):
        prompt = to_pt(prompt).cuda()
        response = to_pt(response).cuda()
        input_ids.append(torch.cat([prompt, response]))
        position_ids.append(torch.arange(len(prompt) + len(response)).cuda())
        seqlens.append(len(prompt) + len(response))
        gate_mask.append(F.pad(torch.tensor([1], dtype=torch.bool, device="cuda"), (len(prompt)-5, len(response)+4)))
        if seq_reward:
            gate_idx.append(F.pad(torch.ones(1, dtype=torch.long, device="cuda") * idx, (len(response)+len(prompt)-1, 0), value=-100))
        else:
            gate_idx.append(F.pad(torch.ones(len(response)+1, dtype=torch.long, device="cuda") * idx, (len(prompt)-1, 0), value=-100))

    seqlens = torch.tensor(seqlens, device="cuda")
    max_seqlen = seqlens.max().item()
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).int()
    input_ids = torch.cat(input_ids)[None]
    position_ids = torch.cat(position_ids)[None]
    gate_mask = torch.cat(gate_mask)[None]
    gate_idx = torch.cat(gate_idx)[None]

    assert input_ids[gate_mask].eq(128009).all()
    # assert input_ids[gate_idx != -100].eq(128009).all()

    return {"input_ids": input_ids, 
            "position_ids": position_ids,
            "gate_mask": gate_mask, 
            "gate_idx": gate_idx,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,}

def do_batch(input_ids, position_ids, score_mask, gate_mask, start_id_q, end_id_k):
    seqlens = torch.tensor([len(ex) for ex in input_ids], device="cuda")
    max_seqlen = seqlens.max().item()
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).int()
    input_ids = to_pt(list(chain(*input_ids))).cuda()[None]
    position_ids = to_pt(list(chain(*position_ids))).cuda()[None]
    gate_mask = to_pt(list(chain(*gate_mask))).cuda().bool()[None]
    gate_idx = torch.cat([torch.where(torch.tensor(mask, device="cuda", dtype=torch.bool), idx, -100) for idx, mask in enumerate(score_mask)])[None]
    end_id_k = to_pt(list(chain(*end_id_k))).cuda().int()
    start_id_q = to_pt(list(chain(*start_id_q))).cuda().int()

    # token_pattern = tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False, )
    token_pattern = [128009, 128006, 78191, 128007, 271]
    idx = torch.arange(0, len(gate_mask[0]), device="cuda")[gate_mask[0]]
    for i in range(len(token_pattern)):
        assert input_ids[0][idx + i].eq(token_pattern[i]).all()
    assert input_ids[gate_idx != -100].eq(128009).all()

    return {"input_ids": input_ids, 
            "position_ids": position_ids,
            "gate_mask": gate_mask, 
            "gate_idx": gate_idx,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,
            "end_id_k": end_id_k, 
            "start_id_q": start_id_q}


class GatingNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, temperature: float = 10,
                 logit_scale: float = 1., hidden_dim: int = 1024, n_hidden: int = 3):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        # Apply the conditional ReLU using the expanded mask
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]


@dataclass
class CustomOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        hidden_state (`Tuple[torch.FloatTensor]` of length `config.num_hidden_layers`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        prompt_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The embeddings of the prompt tokens.
        gating_output (`torch.FloatTensor` of shape `(batch_size, config.num_objectives)`):
            The logits for the gating network.
        score (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            The final reward score.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Same as score
    """

    rewards: torch.FloatTensor = None
    hidden_state: Optional[Tuple[torch.FloatTensor, ...]] = None
    prompt_embedding: Optional[torch.FloatTensor] = None
    gating_output: Optional[torch.FloatTensor] = None
    score: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LlamaForRewardModelWithGating(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        config_dict = config.to_dict()
        self.num_objectives = config_dict.get("num_objectives", 19)
        self.regression_layer = nn.Linear(config.hidden_size, self.num_objectives, bias=False)
        self.post_init()
        # Not using torch.eye because it is not supported in BF16
        I = torch.zeros(self.num_objectives, self.num_objectives)
        I[range(self.num_objectives), range(self.num_objectives)] = 1.
        self.reward_transform_matrix = nn.Parameter(I)
        self.reward_transform_matrix.requires_grad = False

        # Initialize weights and apply final processing
        self.gating = GatingNetwork(config.hidden_size, config.num_objectives,
                                    temperature=config_dict.get("gating_temperature", 10),
                                    hidden_dim=config_dict.get("gating_hidden_dim", 1024),
                                    n_hidden=config_dict.get("gating_n_hidden", 3))

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        gate_mask = kwargs.pop("gate_mask", None)
        gate_pos = kwargs.pop("gate_pos", None)
        assert gate_mask is not None or gate_pos is not None
        gate_idx = kwargs.pop("gate_idx")

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        tokens_hidden_states = transformer_outputs[0]

        hidden_states = tokens_hidden_states[gate_idx != -100]
        _gate_idx = gate_idx[gate_idx != -100]
        rewards = self.regression_layer(hidden_states)

        if gate_mask is not None:
            gate_embedding = tokens_hidden_states[gate_mask]
        else:
            gate_embedding = tokens_hidden_states[gate_pos]
        gating_output = self.gating(gate_embedding)

        rewards_adjusted = rewards @ self.reward_transform_matrix
        score = torch.sum(gating_output[_gate_idx] * rewards_adjusted, dim=1)

        return CustomOutput(
            rewards=rewards,
            hidden_state=hidden_states,
            prompt_embedding=gate_embedding,
            gating_output=gating_output,
            score=score,
            logits=score,
        )