set -e
output_dir=""
checkpoints=""
test_dirs="./test_data"
test_files=""
prompt_type="completion"
seed="0"
only_score=0

while true; do
    case "$1" in
        --output_dir) output_dir="$2"; shift 2 ;;
        --checkpoints) checkpoints="$2"; shift 2 ;;
        --test_dirs) test_dirs="$2"; shift 2 ;;
        --test_files) test_files="$2"; shift 2 ;;
        --prompt_type) prompt_type="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --only_score) only_score="$2"; shift 2 ;;
        "") break ;;
        *) echo "Internal error!<e>$1<e>"; exit 1 ;;
    esac
done

if [ "${checkpoints}" == "" ]; then
    checkpoints=$(ls -l ${output_dir} | grep " checkpoint" | awk '{print  $NF}')
fi

for checkpoint in ${checkpoints[@]}; do
    ck_dir=${output_dir}/${checkpoint}
    mkdir -p ${output_dir}/test_result-${checkpoint}
    if [ -f "${ck_dir}/adapter_config.json" ]; then
        peft_model="--peft_model "${ck_dir}
        base_model=$(jq -r '.base_model_name_or_path' ${ck_dir}/adapter_config.json)
    else
        if [ -f "${ck_dir}/config.json" ]; then
            peft_model=""
            base_model=${ck_dir}
        else
            echo "no config in "$ck_dir
            continue
        fi
    fi
    
    if [[ "$only_score" != "true" && "$only_score" != "True" && "$only_score" != "1" ]]; then
        python ./eval/get_output_fast.py \
            --model "${base_model}" \
            ${peft_model} \
            --test_dirs "${test_dirs}" \
            --output_dir ${output_dir}/test_result-${checkpoint} \
            --test_files "${test_files}" \
            --prompt_type "${prompt_type}" \
            --seed "${seed}" \
            >${output_dir}/test_result-${checkpoint}.log 2>&1 
    fi

    python ./eval/get_stats_fast.py \
        --test_dir  "${test_dirs}" \
        --output_dir ${output_dir}/test_result-${checkpoint} \
        --test_files "${test_files}" \
        --seed "${seed}" \
        >${output_dir}/test_stats-${checkpoint}.log 2>&1 
done