export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_NAME="llama-3-8b-it"
export DATA_NAME="uf.fine-grained.${MODEL_NAME}_ids"
export PORT=30001

name=get_ref
bash ./runs/$name.sh outputs/$name
