LORA_CONFIG='{"peft_type": "LORA_MOE", "r": 16, "num_gates": 8, "use_dora": false, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj", "down_proj"], "lora_dropout": 0}'

mkdir -p $1

OMP_NUM_THREADS=8 accelerate launch --main_process_port "$PORT" --config_file ./configs/deepspeed_4gpus_stage0.yaml \
     ./train.py \
    --model_id ./models/$MODEL_NAME \
    --use_peft 1 \
    --use_lora_moe 1 \
    --peft_config "$LORA_CONFIG" \
    --with_efficient_module optimized_module \
    --gradient_checkpointing ${gradient_checkpointing:-1} \
    --torch_dtype bfloat16 \
    --bf16 \
    --formated_train_data_cache_path ./cache_ids/${DATA_NAME}.json.gz \
    --prompt_type completion \
    --remove_unused_columns 0 \
    --do_train \
    --check_stage no_ck \
    --training_type sft \
    --optim adamw_torch \
    --learning_rate $LR \
    --weight_decay 0 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --per_deivce_train_batch_max_tokens $per_deivce_train_batch_max_tokens \
    --batch_tokens_divider $batch_tokens_divider \
    --h_accumulation_steps $h_accumulation_steps \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 1 \
    --eval_strategy no \
    --eval_steps 0.1 \
    --save_strategy epoch \
    --save_only_model 1 \
    --greater_is_better False \
    --logging_strategy steps \
    --logging_steps 1 \
    --include_tokens_per_second \
    --output_dir $1 \
    --num_train_epochs $num_epochs \
    --max_length 4096 \
    --seed 0 \
    --run_name $(basename $1) \
    --report_to wandb \
    > $1/training.log 2>&1 