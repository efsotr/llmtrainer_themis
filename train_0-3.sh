set -e
export WANDB_PROJECT="PO_Try"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT=30003

export MODEL_NAME="llama-3-8b-it"
export DATA_NAME="uf."${MODEL_NAME}"_4r.armo."${MODEL_NAME}"_ids"
# export per_deivce_train_batch_max_tokens=10496
export per_deivce_train_batch_max_tokens=15040
export batch_tokens_divider=2
export h_accumulation_steps=2

train() {
    bash ./runs/$1.sh outputs/$2
}

eval() {
    bash ./eval/eval.sh --output_dir outputs/$1 
}

score() {
    bash ./eval/eval.sh --output_dir outputs/$1 --only_score 1
}

export PO_BETA=0.01
export LR=1e-6
name=dpo
train $name llama3-llamam3-8b-armo-$name
eval llama3-llamam3-8b-armo-$name
