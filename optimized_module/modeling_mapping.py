from .modeling_gemma2_multi_layers_ckpt import Gemma2ForCausalLM, Gemma2Model
from .modeling_qwen2_multi_layers_ckpt import Qwen2ForCausalLM, Qwen2Model
from .modeling_llama_multi_layers_ckpt import LlamaForCausalLM, LlamaModel
from .modeling_mistral_multi_layers_ckpt import MistralForCausalLM, MistralModel

from transformers import LlamaConfig, Qwen2Config, Gemma2Config, MistralConfig

CAUSAL_MODEL_MAPPING = {
    LlamaConfig.model_type: LlamaForCausalLM, 
    Qwen2Config.model_type: Qwen2ForCausalLM, 
    Gemma2Config.model_type: Gemma2ForCausalLM,
    MistralConfig.model_type: MistralForCausalLM,
}

MODEL_MAPPING = {
    LlamaConfig.model_type: LlamaModel, 
    Qwen2Config.model_type: Qwen2Model, 
    Gemma2Config.model_type: Gemma2Model,
    MistralConfig.model_type: MistralModel,
}