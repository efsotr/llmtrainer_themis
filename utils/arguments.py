import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, Dict

import torch
from transformers.trainer import TrainingArguments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ModelArguments:
    model_id : Optional[str] = field(
        default=None, 
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    ref_model_id: Optional[str] = field(default=None)
    reward_model_id : Optional[str] = field(default=None)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    tokenizer_id: Optional[str] = field(
        default=None
    ) 
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer."}
    )
    torch_dtype: str = field(
        default="float32",
        metadata={"help": "Override the dtype of model.",
                  "choices" : ["float16", "bfloat16", "float32"]}
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether use PEFT (Parameter Efficient Fine-Tuning)."}
    )
    peft_config: str = field(
        default='{"peft_type": "LORA", "r": 16, "lora_alpha": 32, "target_modules": ["down_proj"], "lora_dropout": 0.05}',
        metadata={"help": "The config for PEFT (e.g. Lora)."}
    )
    use_lora_moe: bool = field(
        default=False
    )
    use_lora_moe2: bool = field(
        default=False
    )
    peft_model_id: str = field(
        default=None,
        metadata={"help": "The peft model checkpoint for peft weights initialization."}
    )
    attn_implementation: str = field(
        default='flash_attention_2',
        metadata={"help": "The attention implementation.",
                  "choices": ["eager", "flash_attention_2"]}
    )

    def __post_init__(self):
        if self.tokenizer_id is None:
            self.tokenizer_id = self.model_id
        self.torch_dtype = getattr(torch, self.torch_dtype)
        if self.use_lora_moe:
            assert self.use_peft

@dataclass
class DataTrainingArguments:
    sample_ratio: Optional[float] = field(
        default=1, metadata={"help": "The training of sample ratio."}
    )
    max_prompt_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total prompt sequence length text after tokenization."},
    ) 
    max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total sequence length text after tokenization."},
    )
    formated_train_data_cache_path: str = field(
        default=None
    )
    formated_dev_data_cache_path: str = field(
        default=None
    )
    ref_cache_path: Optional[str] = field(default=None)
    adv_cache_path: Optional[str] = field(default=None)
    train_extra_args: Optional[str] = field(default=None)
    valid_extra_args: Optional[str] = field(default=None)
    prompt_type: str = field(default="completion", 
                             metadata={"choices": ["completion", "chat"]})
    chat_template : str = field(default=None)

    def __post_init__(self):
        self.train_extra_args = self.train_extra_args.split(',') if self.train_extra_args else []
        self.valid_extra_args = self.valid_extra_args.split(',') if self.valid_extra_args else []

@dataclass
class MaxTrainingArguments(TrainingArguments):
    eos_token_id: int = field(default=None)
    pad_token_id: int = field(default=0)

    with_efficient_module: str = field(
        default="",
        metadata={"help": "Whether use efficient module.", 
                  "choices": ["", "optimized_module"]},
    )
    gradient_checkpointing_layers: int = field(default=2)

    h_accumulation_steps: int = field(default=1)
    per_deivce_train_batch_max_tokens: int = field(default=0)
    batch_tokens_divider: int = field(default=1) 

    check_stage: str = field(
        default="no_ck", 
        metadata={"choices": ["no_ck", "ck_data", "ck_ref", "ck_run"]}
    )

    training_type: str = field(
        default=None,
        metadata={"help": "training type", 
                  "choices": ["sft", "po", "vm", "rm", "xpo", "ord"]}
    )

    ## PO
    norm_by_len: bool = field(default=False)
    with_ref: bool = field(default=False)
    po_beta: float = field(default=0)
    po_gamma: float = field(default=0)
    norm_by_len_plus_one: float = field(default=0.)
    use_sampo: float = field(default=False)
    enable_norm_win_lose: bool = field(default=False)

    ## RM
    rm_beta: float = field(default=1)
    rm_gamma: float = field(default=0)
    value_head_learning_rate: float = field(default=0)
    value_head_zero_init: bool = field(default=False)

    ## Ord
    cutpoints_learning_rate: float = field(default=0)
    ord_num_labels: int = field(default=0)

    ## 
    extra_save_steps: int = field(default=None)

    ##
    log_mem: bool = field(default=False)
    
    def __post_init__(self):
        super().__post_init__()

        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {}
        self.gradient_checkpointing_kwargs["use_reentrant"] = True

        if self.value_head_learning_rate == 0:
            self.value_head_learning_rate = self.learning_rate
        if self.cutpoints_learning_rate == 0:
            self.cutpoints_learning_rate = self.value_head_learning_rate
