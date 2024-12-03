MODEL_NAME="llama-3-8b"
DATA_NAME="eval_train_data_v59_a"
python get_tokenized_data.py \
    --model_path ./models/$MODEL_NAME \
    --prompt_type completion \
    --training_type sft \
    --train_dataset_path train_data/$DATA_NAME.json \
    --train_save_path ./cache_ids/$DATA_NAME.${MODEL_NAME}_ids.json.gz \