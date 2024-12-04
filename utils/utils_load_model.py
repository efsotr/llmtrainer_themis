# coding=utf-8

import os
import json
import logging
from typing import cast

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer, 
    PretrainedConfig
)
from peft import PeftConfig, PeftModelForCausalLM

from . import get_data_args, get_model_args, get_training_args
from .arguments_init import set_tokenizer
from optimized_module.wrap_efficient_module import replace_with_efficient_module
from tokenize_ids import get_eos

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def load_tokenizer():
    model_args = get_model_args()
    data_args = get_data_args()
    training_args = get_training_args()

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "add_eos_token": False,
    }
        
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_id, **tokenizer_kwargs)

    if data_args.prompt_type == "chat" and data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template

    tokenizer.eos_token_id, eos_token = get_eos(tokenizer, data_args.prompt_type)
    training_args.eos_token_id = tokenizer.eos_token_id
    logger.info(f"eos token: {eos_token}")

    set_tokenizer(tokenizer)
    return tokenizer

def load_model():
    model_args = get_model_args()
    training_args = get_training_args()

    config_kwargs = {
        "revision": model_args.model_revision,
        "torch_dtype" : model_args.torch_dtype,
        "use_cache" : False,
        "attn_implementation" : model_args.attn_implementation,
    }

    config = cast(PretrainedConfig, AutoConfig.from_pretrained(model_args.model_id, **config_kwargs))
    config.gradient_checkpointing_layers = training_args.gradient_checkpointing_layers
    config.initializer_range = config.hidden_size ** -0.5
    config.use_tree_attn = True
    
    if training_args.training_type in ["rm"]:
        from optimized_module.value_model import ValueModel
        model : ValueModel = ValueModel.from_pretrained(
            model_args.model_id,
            config=config,
            torch_dtype=config.torch_dtype,
            device_map="cpu"
        )
        if training_args.value_head_zero_init:
            model.score.weight.data.zero_()
    elif training_args.training_type in ["ord"]:
        from optimized_module.value_model import OrdinalRegressionModel
        config.num_labels = training_args.ord_num_labels
        model : OrdinalRegressionModel = OrdinalRegressionModel.from_pretrained(
            model_args.model_id,
            config=config,
            torch_dtype=config.torch_dtype,
            device_map="cpu"
        )
        if training_args.value_head_zero_init:
            model.score.weight.data.zero_()
        model.cutpoints.data = torch.arange(0, config.num_labels - 1) - (config.num_labels / 2 - 1)
        model.inf.data = torch.tensor([float('inf')])
    else:
        from optimized_module import CAUSAL_MODEL_MAPPING
        model = CAUSAL_MODEL_MAPPING[config.model_type].from_pretrained(
            model_args.model_id,
            config=config,
            torch_dtype=config.torch_dtype,
            device_map="cpu"
        )
    
    model.cuda()
    if model_args.use_peft:
        if model_args.peft_model_id:
            assert model_args.use_lora_moe is False
            model = PeftModelForCausalLM.from_pretrained(model, model_args.peft_model_id, is_trainable=True)
            peft_config = cast(PeftConfig, model.peft_config)
        else:
            if model_args.peft_config.strip()[0] == '{':
                model_args.peft_config = json.loads(model_args.peft_config)
            elif os.path.isfile(model_args.peft_config):
                model_args.peft_config = json.load(open(model_args.peft_config))
            else:
                raise NotImplementedError(model_args.peft_config)
            model_args.peft_config.update({"task_type": "CAUSAL_LM"})
            peft_config = PeftConfig(**model_args.peft_config)
            model = PeftModelForCausalLM(model, peft_config)
            model.peft_config['default'].peft_type = "LORA" # No implementation provided for save_pretrained, reverting to LORA

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        print_trainable_parameters(model)

    if training_args.with_efficient_module == "optimized_module":
        replace_with_efficient_module(model)

    if training_args.gradient_checkpointing:
        from optimized_module.recompute import set_recompute_flag
        set_recompute_flag(False)

    return model
