set -e
export WANDB_PROJECT="Themis"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PORT=30002

export MODEL_NAME="llama-3-8b"
export DATA_NAME="eval_train_data_v59_a."${MODEL_NAME}"_ids"
export per_deivce_train_batch_max_tokens=7936
export batch_tokens_divider=1
export h_accumulation_steps=2

# 7936 * 2

train() {
    bash ./runs/$1.sh outputs/$2
}

eval() {
    bash ./eval/eval.sh --output_dir outputs/$1 
}

score() {
    bash ./eval/eval.sh --output_dir outputs/$1 --only_score 1
}

export gradient_checkpointing=0
export num_epochs=3
export LR=1e-3
name=lora
exper_name=llama3-$name-LR-$LR
# train $name $exper_name
eval $exper_name

export gradient_checkpointing=0
export num_epochs=3
export LR=2e-3
name=lora
exper_name=llama3-$name-LR-$LR
train $name $exper_name
eval $exper_name


export gradient_checkpointing=0
export num_epochs=3
export LR=5e-4
name=lora
exper_name=llama3-$name-LR-$LR
train $name $exper_name
eval $exper_name

export gradient_checkpointing=0
export num_epochs=3
export LR=8e-4
name=lora
exper_name=llama3-$name-LR-$LR
train $name $exper_name
eval $exper_name
