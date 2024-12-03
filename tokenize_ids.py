from typing import Dict
from transformers import LlamaTokenizer, LlamaTokenizerFast

PROMPT_W_ADD = "###Instruction###\n\
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.\n\
Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n\
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n\
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.\n\
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.\n\
\n\
###Evaluation Criterion###\n\
{aspect_des}\n\
\n\
###Example###\n\
{source_des}:\n\
{source}\n\
\n\
{addition_des}:\n\
{addition}\n\
\n\
{target_des}:\n\
{target}\n\
\n\
###Your Evaluation###\n"

PROMPT_WO_ADD = "###Instruction###\n\
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.\n\
Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n\
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n\
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.\n\
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.\n\
\n\
###Evaluation Criterion###\n\
{aspect_des}\n\
\n\
###Example###\n\
{source_des}:\n\
{source}\n\
\n\
{target_des}:\n\
{target}\n\
\n\
###Your Evaluation###\n"

def get_prompt(ex):
    return PROMPT_WO_ADD if ex.get("addition", None) is None else PROMPT_W_ADD

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

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
    if system_prompt is None:
        system_prompt = ex.get("system_prompt", None)

    if prompt_type == "completion":
        prompt = get_prompt(ex).format_map({**ex, "system_prompt": system_prompt})
        if not tokenize:
            return prompt
        return tokenizer.encode(prompt)

    if prompt_type == "chat":
        conversation = []
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt.strip()})
        conversation.append({"role": "user", "content": get_prompt(ex).format_map(**ex)})
        prompt = tokenizer.apply_chat_template(conversation, 
                                               tokenize=tokenize, 
                                               add_generation_prompt=True)
        return prompt
    
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