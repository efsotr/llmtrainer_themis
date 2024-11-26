set -e

check_variable() {
    local var_name=$1
    local var_value=${!var_name}

    if [ -z "${var_value+x}" ]; then
        echo "Error: $var_name is not set." >&2
        return 1  # 返回非零表示错误
    else
        echo "$var_name is set to: $var_value"
    fi
}

MODEL_DIR={$MODEL_DIR:-"./models/"}
CHECKPOINT={$CHECKPOINT:-""}

echo ""
check_variable "MODEL_DIR"
check_variable "MODEL_NAME"
check_variable "CHECKPOINT"
check_variable "DATA_NAME"
check_variable "PORT"

mkdir -p $1

accelerate launch --main_process_port "$PORT" --config_file ./configs/deepspeed_xgpus_stage0.yaml \
     get_logits.py \
    --model_id $MODEL_DIR/$MODEL_NAME/$CHECKPOINT \
    --with_efficient_module optimized_module \
    --torch_dtype bfloat16 \
    --bf16 \
    --formated_train_data_cache_path ./cache_ids/${DATA_NAME}.train.json.gz \
    --formated_dev_data_cache_path ./cache_ids/${DATA_NAME}.dev.json \
    --ref_cache_path ./cache_logits/${DATA_NAME}.${MODEL_NAME}_logits.pt.gz \
    --prompt_type chat \
    --do_train \
    --do_eval \
    --training_type po \
    --per_deivce_train_batch_max_tokens 15360 \
    --batch_tokens_divider 1 \
    --include_tokens_per_second \
    --output_dir $1 \
    --max_length 4096 \
    --seed 0 \
    > $1/training.log 2>&1 
