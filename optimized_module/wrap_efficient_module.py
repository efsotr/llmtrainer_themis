import logging
import transformers.utils.logging

import torch
from torch import nn
from transformers import PreTrainedModel

from .layer_norm import EffRMSNorm, EffLayerNorm
from .mlp import EffActMLP, ACT2FN
from .modeling_llama_multi_layers_ckpt import LlamaRMSNorm
from .modeling_gemma2_multi_layers_ckpt import Gemma2RMSNorm
from .modeling_qwen2_multi_layers_ckpt import Qwen2RMSNorm
from .modeling_mistral_multi_layers_ckpt import MistralRMSNorm
from .dropout import EffDropout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# RMSNorm(x) = (x / sum(x * x)) * w
# MLP(x) = Down(Act(Gate(x)) * Up(x))

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def replace_with_efficient_module(model : PreTrainedModel, replace_norm_type = True, replace_mlp_act_type = True):
    logger.info(f"Replace Norm type {replace_norm_type}")
    logger.info(f"Replace MLP Act type {replace_mlp_act_type}")

    replaced_mlp_act_modules = []
    replaced_norm_modules = []
    replaced_dropout = []

    for name, _ in model.named_modules():
        parent, target, target_name = _get_submodules(model, name)

        if isinstance(target, nn.LayerNorm):
            if replace_norm_type is None:
                continue
            logger.warning_once("LayerNorm")
            replaced_norm_modules.append(name)
            new_module = EffLayerNorm.from_ori(target)
            setattr(parent, target_name, new_module)
            del target

        elif isinstance(target, (LlamaRMSNorm, Qwen2RMSNorm, MistralRMSNorm)):
            if replace_norm_type is None:
                continue
            logger.warning_once("LlamaRMSNorm/Qwen2RMSNorm/MistralRMSNorm")
            replaced_norm_modules.append(name)
            new_module = EffRMSNorm.from_ori(target)
            setattr(parent, target_name, new_module)
            del target

        elif isinstance(target, Gemma2RMSNorm):
            if replace_norm_type is None:
                continue
            logger.warning_once("Gemma2RMSNorm")
            replaced_norm_modules.append(name)
            new_module = EffRMSNorm.from_ori(target, weight_add_one=True)
            setattr(parent, target_name, new_module)
            del target

        elif target_name == "mlp":
            if replace_mlp_act_type is None:
                continue
            if target.config.hidden_act not in ACT2FN.keys():
                logger.warning_once(f"Act_fn of MLP not in {ACT2FN.keys()}.")
                continue

            logger.warning_once("MLP")
            replaced_mlp_act_modules.append(name)

            new_module = EffActMLP.from_ori(target)
            setattr(parent, target_name, new_module)
            del target
            
        elif isinstance(target, nn.Dropout) and "lora_dropout" in name:
            logger.warning_once("LoRA Dropout")
            replaced_dropout.append(name)
            new_module = EffDropout.from_ori(target)
            setattr(parent, target_name, new_module)
            del target

    if replace_mlp_act_type is not None:
        logger.info(f"replaced mlp act: {replaced_mlp_act_modules}")
    if replace_norm_type is not None:
        logger.info(f"replaced norm: {replaced_norm_modules}")
    if len(replaced_dropout) > 0:
        logger.info(f"replaced dropout: {replaced_dropout}")
    