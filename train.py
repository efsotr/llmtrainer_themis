# coding=utf-8

import sys
import logging
import importlib

import transformers

from file_IO import safe_load
from utils import (
    init_args, get_training_args, get_data_args, get_tokenizer,
    preprocess_data, EndEvalCallback, 
    load_model, load_tokenizer, 
    load_logits
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    init_args()

    training_args = get_training_args()

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # load tokenizer
    load_tokenizer()

    # Get the datasets 
    def get_data():
        data_args = get_data_args()
        if "," not in data_args.formated_train_data_cache_path:
            train_formated_data = safe_load(data_args.formated_train_data_cache_path)
        else:
            paths = data_args.formated_train_data_cache_path.split(",")
            train_formated_data = {}
            cnt = 0
            for path in paths:
                data = safe_load(path)
                assert isinstance(data, dict)
                for key in data.keys():
                    train_formated_data[f"{key}_train{cnt}"] = data[key]
                cnt += 1

        if data_args.formated_dev_data_cache_path is not None:
            if "," not in data_args.formated_train_data_cache_path:
                dev_formated_data = safe_load(data_args.formated_dev_data_cache_path)
            else:
                paths = data_args.formated_dev_data_cache_path.split(",")
                dev_formated_data = {}
                cnt = 0
                for path in paths:
                    data = safe_load(path)
                    assert isinstance(data, dict)
                    for key in data.keys():
                        dev_formated_data[f"{key}_dev{cnt}"] = data[key]
                    cnt += 1

        return preprocess_data((train_formated_data, dev_formated_data))
    
    train_datasets, eval_datasets, collate_fn = get_data()
    logger.info('Finish loading dataset')

    if training_args.check_stage == "ck_data":
        return

    load_logits(train_datasets, eval_datasets)

    if training_args.check_stage == "ck_ref":
        return

    # Load model
    model = load_model()

    module = importlib.import_module("utils.trainer_" + training_args.training_type)
    trainer_class = module.Trainer
    metric_class = module.Metric

    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        tokenizer=get_tokenizer(),
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=metric_class(), 
        callbacks=[EndEvalCallback],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()

