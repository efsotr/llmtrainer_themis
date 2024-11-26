from typing import Dict
from transformers import LlamaTokenizer, LlamaTokenizerFast

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
PROMPT = """{prompt}\n\n"""

def get_token_ids(text: str, tokenizer):
    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        ids = tokenizer.encode("\n" + text, add_special_tokens=False)[2:] 
    else:
        ids =  tokenizer.encode(tokenizer.eos_token + text, add_special_tokens=False)[1:]
    return ids

def get_prompt_ids(
    ex: Dict[int, str], 
    tokenizer: LlamaTokenizer, 
    prompt_type: str = "chat", 
    tokenize = True, 
    system_prompt : str = None
):
    # one turn conversation
    response_prefix = ex.get("prefix", "")
    if system_prompt is None:
        system_prompt = ex.get("system_prompt", None)

    if prompt_type == "completion":
        prompt = PROMPT.format_map({**ex, "system_prompt": system_prompt}) + response_prefix
        if not tokenize:
            return prompt
        return tokenizer.encode(prompt)

    if prompt_type == "chat":
        conversation = []
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt.strip()})
        conversation.append({"role": "user", "content": ex["prompt"].strip()})
        prompt = tokenizer.apply_chat_template(conversation, 
                                               tokenize=tokenize, 
                                               add_generation_prompt=True)
        prefix = get_token_ids(response_prefix, tokenizer)
        return prompt + prefix
    
    raise NotImplementedError

def get_eos(tokenizer: LlamaTokenizer, prompt_type: str = "chat") -> tuple[int, str]:
    if prompt_type == "chat":
        ex = tokenizer.apply_chat_template([{"role": "user", "content":"ok"}, 
                                            {"role": "assistant", "content":"ok"}], 
                                            tokenize=False).strip()
        eos_token_id = tokenizer.encode(ex)[-1]
    elif prompt_type == "completion":
        eos_token_id = tokenizer.eos_token_id 
    else:
        raise NotImplementedError
    
    eos_token = tokenizer._convert_id_to_token(eos_token_id)
    return eos_token_id, eos_token