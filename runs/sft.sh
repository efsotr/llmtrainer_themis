mkdir -p $1

OMP_NUM_THREADS=8 accelerate launch --main_process_port "$PORT" --config_file ./configs/deepspeed_4gpus_stage2_off_o.yaml \
     train.py \
    --model_id /data/pretrain/Qwen/Qwen2-7B-Instruct \
    --with_efficient_module optimized_module \
    --gradient_checkpointing 1 \
    --torch_dtype bfloat16 \
    --bf16 \
    --formated_train_data_cache_path $train_tokenids_path \
    --formated_dev_data_cache_path $dev_tokenids_path \
    --prompt_type completion \
    --remove_unused_columns 0 \
    --do_train \
    --do_eval \
    --check_stage no_ck \
    --training_type sft \
    --optim adamw_torch \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --per_deivce_train_batch_max_tokens 7680 \
    --batch_tokens_divider 1 \
    --h_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
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
    --max_length $max_length \
    --seed 0 \
    --run_name $(basename $1) \
    --report_to none \
    > $1/training.log 2>&1 
