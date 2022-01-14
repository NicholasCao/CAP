import argparse

import torch

from data_utils.processors import DataProcessor, PROCESSORS
from src.config import WrapperConfig
from src.wrapper import TransformerModelWrapper
from src.utils import Config, set_seed

import log

logger = log.get_logger('root')

default_config = {
    "num_train_epochs": 3,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "learning_rate": 5e-5,
    "warmup_steps": 0,
    "gradient_accumulation_steps": 1,
    "pattern_id": 0,
    "seed": 256,
    "wrapper_type": "rtd",

    "model_type": "deberta",
    "model_name_or_path": "microsoft/deberta-v3-base",
    "num_train_example": -1,
    "unique_prompt": True,

    "max_grad_norm": 1.0,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.01,
    "max_seq_length": 256,
    "no_cuda": False,
    "do_train": True,
    "do_eval": True,
    "eval_every_step": 100,
    "hidden_size": 256,
    "zero_shot": False,
    "output_dir": "output",
    "save_model": False,
    "base_model_learning_rate": None
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path of config file")

    args = parser.parse_args()
    config_path = args.config

    config = Config()
    config.read_config(config_path)
    config.insert_default(default_config)

    logger.info("Parameters: {}".format(config))

    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"

    processor: DataProcessor = PROCESSORS[config.task_name](config.data_dir, log_example=True)

    train_examples = processor.get_train_examples(num_example=config.num_train_example)
    dev_examples = processor.get_dev_examples()

    label_list = processor.get_labels()

    wrapper_config = WrapperConfig(
        wrapper_type=config.wrapper_type,
        model_type=config.model_type,
        model_name_or_path=config.model_name_or_path,
        task_name=config.task_name,
        label_list=label_list,
        device=device,
        pattern_id=config.pattern_id,
        max_seq_length=config.max_seq_length,
        output_dir=config.output_dir,
        hidden_size=config.hidden_size,
        unique_prompt=config.unique_prompt,
        zero_shot=config.zero_shot,
        save_model=config.save_model)

    wrapper = TransformerModelWrapper(wrapper_config)

    wrapper.train(
        train_data=train_examples,
        train_batch_size=config.train_batch_size,
        eval_data=dev_examples,
        eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_epsilon=config.adam_epsilon,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        eval_every_step=config.eval_every_step,
        base_model_learning_rate=config.base_model_learning_rate)

if __name__ == "__main__":
    main()
