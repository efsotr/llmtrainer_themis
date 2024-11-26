set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_NAME=llama-3-8b-it
python ./eval/get_output.py \
    --model ./models/$MODEL_NAME \
    --batch_size 64 \
    --test_dirs ./train_data/ori_ultrafeedback \
    --test_files ori-uf.dev.json \
    --output_dir ./train_data/$MODEL_NAME \
    --sampling_params ./eval/gen_4r.json \
    --prompt_type chat \
    --cache_dir ./cache/dev_cache 
    
python ./eval/get_output.py \
    --model ./models/$MODEL_NAME \
    --batch_size 128 \
    --test_dirs ./train_data/ori_ultrafeedback \
    --test_files ori-uf.train.json.gz \
    --output_dir ./train_data/$MODEL_NAME \
    --sampling_params ./eval/gen_4r.json \
    --prompt_type chat \
    --cache_dir ./cache/train_cache 

python ./eval/get_stats.py \
    --test_dirs train_data/ori_ultrafeedback \
    --test_files ori-uf.dev.json,ori-uf.train.json.gz \
    --output_dir train_data/$MODEL_NAME 

bash train_4-7.sh &
bash train_0-3.sh &