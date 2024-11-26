# llmtrainer

## 环境

看 https://zhuanlan.zhihu.com/p/540615230 ，打包好的环境在34服务器的/home/linli/LLM.tar.gz

## 数据集准备

### 数据集组织格式
数据集内数据组织格式：
sft: List[{"prompt_id": "", "prompt": "Your prompt", "response": "Your response"}]
dpo: List[{"prompt_id": "", "prompt": "Your prompt", "chosen": "Your response", "rejected": "Your response"}]

+ 注意1：如果不不使用instruct model，那么需要注意prompt 和 response 是 tokenized 后直接拼在一起的，所以最好 prompt 是以 `\n` 为结尾。
+ 注意2：文件的组织结构是通过识别文件后缀获取的。目前仅支持 json, jsonl, pt 文件类型（以及它们的 gzip 压缩版本，比如 json.gz ）。

### tokenize 数据集
```sh
python ./tokenized_inputs/get_tokenized_data.py \
 --model $model_path \
 --train_dataset_path $train_dataset_path \
 --train_save_path $train_tokenids_path \
 --dev_dataset_path $dev_dataset_path \
 --dev_save_path $dev_tokenids_path \
 --training_type sft \
 --prompt_type completion
```
注意1：如果只想 tokenize train dataset，那么`--dev_dataset_path $dev_dataset_path`这一行可以去掉，并且只想 tokenize dev dataset 同理。

## 训练准备

训练的启动脚本参考：`train.sh`
需要指定写好的训练设置脚本，例如 `runs/sft.sh`。并且需要用像`export CUDA_VISIBLE_DEVICES=0,1`来指定用哪些 gpu。

如果要测试，需要指定 `test_dataset_dir` (会将 `test_dataset_dir` 下所有文件视为测试集，如果只需要指定部分文件，那么加上 `--test_files $test_files`，其中`test_files`为需要的部分文件的文件名(只要文件名，不需要加任何的路径前缀)用`,`作为拼接符拼接起来的字符串)。 

### 训练设置脚本

具体看`run/dpo.sh`

`config_file` 用来指定 deepspeed 的设置
+ 显存占用 stage0 > stage1 > stage2 > stage2_off_o 
+ 运行速度 stage0 < stage1 < stage2 < stage2_off_o 

`per_deivce_train_batch_max_tokens` 指定一个 gpu 中一个 batch 最多的 token 数目
+ 为了实现与普通的只设置 `batch_size` 的方法对齐，请运行`python auto_get_size.py $train_tokenids_path $max_length sft`，然后命令行上输入两个用空格分隔的整数 `group_size` 和 `batch_size`，其中 `group_size`=`num_of_gpus`\*`h_accumulation_steps`\*`batch_tokens_divider`。
    + 可以多轮交互
    + 输出为`batch_max_tokens`和`[min_batch_size, max_batch_size]`
    + 然后将`per_deivce_train_batch_max_tokens`设置为`batch_max_tokens`\*`batch_tokens_divider`，所以可以将`batch_tokens_divider`设为大于1的整数使得`max_batch_size - min_batch_size`尽量的小，但需要注意`batch_max_tokens`必须不小于`max_length`。

### 训练启动脚本

对于dpo，需要先计算$\pi_\text{ref}$，具体看`get_logits.sh`以及对应的`runs/get_ref.sh`
具体看`train_0-3.sh`