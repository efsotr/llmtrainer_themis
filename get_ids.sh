export MODEL_NAME="llama-3-8b-it"
DATA_DIR=ultrafeedback_fine-grained
DATA_NAME="uf.fine-grained"
python get_tokenized_data.py \
    --model_path ./models/$MODEL_NAME \
    --prompt_type chat \
    --training_type po \
    --train_dataset_path train_data/$DATA_DIR/$DATA_NAME.train.json.gz \
    --train_save_path ./cache_ids/$DATA_NAME.${MODEL_NAME}_ids.train.json.gz \
    --dev_dataset_path train_data/$DATA_DIR/$DATA_NAME.dev.json \
    --dev_save_path ./cache_ids/$DATA_NAME.${MODEL_NAME}_ids.dev.json \