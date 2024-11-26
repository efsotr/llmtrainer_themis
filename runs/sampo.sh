mkdir -p $1

OMP_NUM_THREADS=8 accelerate launch --main_process_port "$PORT" --config_file ./configs/deepspeed_4gpus_stage2_off_o.yaml \
     ./train.py \
    --model_id ./models/$MODEL_NAME \
    --with_efficient_module optimized_module \
    --gradient_checkpointing ${gradient_checkpointing:-1} \
    --torch_dtype bfloat16 \
    --bf16 \
    --formated_train_data_cache_path ./cache_ids/${DATA_NAME}.train.json.gz \
    --formated_dev_data_cache_path ./cache_ids/${DATA_NAME}.dev.json \
    --ref_cache_path ./cache_logits/${DATA_NAME}.${MODEL_NAME}_logits.pt.gz \
    --prompt_type chat \
    --remove_unused_columns 0 \
    --do_train \
    --do_eval \
    --check_stage no_ck \
    --training_type po \
    --norm_by_len 0 \
    --po_beta $PO_BETA \
    --po_gamma 0 \
    --with_ref 1 \
    --use_sampo 1 \
    --optim adamw_torch \
    --learning_rate $LR \
    --weight_decay 0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --per_deivce_train_batch_max_tokens $per_deivce_train_batch_max_tokens \
    --batch_tokens_divider $batch_tokens_divider \
    --h_accumulation_steps $h_accumulation_steps \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 0.1 \
    --save_strategy epoch \
    --save_only_model 1 \
    --greater_is_better False \
    --logging_strategy steps \
    --logging_steps 1 \
    --include_tokens_per_second \
    --output_dir $1 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --seed 0 \
    --run_name $(basename $1) \
    --report_to wandb \
    > $1/training.log 2>&1 
