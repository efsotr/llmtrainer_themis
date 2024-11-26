from typing import Optional, Union, Tuple, List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .modeling_mapping import MODEL_MAPPING

@dataclass
class CausalLMOutputWithPastValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pred: Optional[Union[torch.FloatTensor, torch.LongTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class ValuePreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer"] ???
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class ValueModel(ValuePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = MODEL_MAPPING[config.model_type](config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict == True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

        hidden_states = outputs[0][labels]
        logits = self.score(hidden_states).squeeze(-1)
        loss = None

        return CausalLMOutputWithPastValue(
            loss=loss,
            logits=logits,
            pred=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class OrdinalRegressionModel(ValuePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = MODEL_MAPPING[config.model_type](config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.cutpoints = nn.Parameter(torch.arange(0, config.num_labels - 1) - (config.num_labels / 2 - 1))
        self.inf = nn.Parameter(torch.tensor([float('inf')]), requires_grad=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.IntTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict == True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

        down, up = labels.chunk(2, dim=-1)
        down = down.squeeze(-1)
        up = up.squeeze(-1)
        hidden_states = outputs[0][down != -100]
        down = down[down != -100]
        up = up[up != -100]
        logits = self.score(hidden_states).squeeze(-1)
        pred = logits.detach()[:, None].ge(self.cutpoints.detach()[None]).sum(-1)
        cutpoints = torch.cat([- self.inf, self.cutpoints, self.inf])
        loss = - torch.clip(F.sigmoid(cutpoints[up] - logits) - F.sigmoid(cutpoints[down] - logits), 1e-9).log().sum()

        return CausalLMOutputWithPastValue(
            loss=loss,
            logits=logits,
            pred=pred,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )