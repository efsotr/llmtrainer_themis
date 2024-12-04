def inject_module():
    from peft import PeftType
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
    from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
    PeftType.LORA_MOE = "LORA_MOE"
    from .config import LoraMoeConfig
    from .model import LoraMoeModel
    PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA_MOE] = LoraMoeModel
    PEFT_TYPE_TO_TUNER_MAPPING[PeftType.LORA_MOE] = LoraMoeModel
    PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA_MOE] = LoraMoeConfig

inject_module()