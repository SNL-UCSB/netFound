import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"To copy construct from a tensor, it is recommended to use sourceTensor",
    category=UserWarning,
)
import math
import random
import os
import torch
import torch.distributed
from dataclasses import field, dataclass
from typing import Optional
from torchinfo import summary

from torch.distributed.elastic.multiprocessing.errors import record
from datasets.distributed import split_dataset_by_node
from transformers import HfArgumentParser, TrainingArguments

from modules.metrics import pretraining_metrics, preprocess_logits_for_metrics
from modules import utils
from modules.samplers import netFoundLengthBucketedIterable
from modules.netFoundDataCollator import DataCollatorWithMeta
from modules.netFoundModels import netFoundLanguageModelling
from modules.netFoundTrainer import netFoundTrainer
from modules.netFoundTokenizer import netFoundTokenizer

random.seed(42)


@dataclass
class PretrainingDataTrainingArguments(utils.CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    no_mlm: bool = field(
        default=None,
        metadata={"help": "no MLM loss function"},
    )
    no_swapped_bursts: bool = field(
        default=None,
        metadata={"help": "no swapped bursts loss function"},
    )
    no_metadata_loss: bool = field(
        default=None,
        metadata={"help": "no metadata loss function"},
    )
    no_direction_loss: bool = field(
        default=None,
        metadata={"help": "no direction loss function"},
    )
    swap_rate: Optional[float] = field(
        default=None,
        metadata={"help": "probability of swapping the burst in the flow during training"},
    )
    subflow_len: Optional[int] = field(
        default=None,
        metadata={"help": "subflow length, -1 for no subflow"},
    )
    mlm_probability: float = field(
        default=None,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    mlm_loss_weight: float = field(
        default=None,
        metadata={"help": "weight for the MLM loss in the total loss"},
    )
    swap_loss_weight: float = field(
        default=None,
        metadata={"help": "weight for the swapped burst loss in the total loss"},
    )
    metadata_loss_weight: float = field(
        default=None,
        metadata={"help": "weight for the metadata loss in the total loss"},
    )
    direction_loss_weight: float = field(
        default=None,
        metadata={"help": "weight for the direction loss in the total loss"},
    )


@record
def main():
    parser = HfArgumentParser(
        (utils.ModelArguments, PretrainingDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    utils.LOGGING_LEVEL = training_args.get_process_log_level()
    logger = utils.get_logger(name=__name__)

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    # Data preparation
    train_dataset, test_dataset = utils.load_train_test_datasets(logger, data_args)
    shuffle_args = {"buffer_size": 10_000} if data_args.streaming else {}
    train_dataset = train_dataset.shuffle(seed=training_args.seed, **shuffle_args)
    if "WORLD_SIZE" in os.environ:
        train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]),
                                              world_size=int(os.environ["WORLD_SIZE"]))
        test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]),
                                             world_size=int(os.environ["WORLD_SIZE"]))

    logger.warning("Tokenizing datasets")
    config = utils.update_config(model_args, data_args, training_args, config=None)
    tokenizer = netFoundTokenizer(config=config)

    if config.no_mlm:
        config.mlm_probability = 0.00001
    if config.no_swapped_bursts:
        config.swap_rate = 0

    data_collator = DataCollatorWithMeta(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
        swap_rate=config.swap_rate
    )

    if "WORLD_SIZE" in os.environ and training_args.local_rank > 0 and not data_args.streaming:
        logger.warning("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    params = {
        "function": tokenizer,
        "batched": True
    }
    if not data_args.streaming:
        params['num_proc'] = data_args.preprocessing_num_workers or utils.get_90_percent_cpu_count()
    train_dataset = train_dataset.map(**params)
    test_dataset = test_dataset.map(**params)

    if "WORLD_SIZE" in os.environ and training_args.local_rank == 0 and not data_args.streaming:
        logger.warning("Loading results from main process")
        torch.distributed.barrier()

    if data_args.streaming and training_args.group_by_length:
        training_args.group_by_length = False
        # only train data to not mess up with batch drops in test
        train_dataset = netFoundLengthBucketedIterable(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            buffer_size=8192,  # tune: 1k–10k usually
            seed=training_args.seed,
            drop_last=True,
        )

    # Model initialization
    if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
    ):
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        model = netFoundLanguageModelling.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        model = netFoundLanguageModelling(config=config)
    model = utils.possibly_freeze(model, model_args)
    if os.environ.get("RANK", "0") == "0":
        summary(model)
    if config.compile:
        model = torch.compile(model, mode="max-autotune")

    training_args.accelerator_config.dispatch_batches = False
    callbacks = []
    if data_args.profile:
        callbacks.append(utils.TorchTBProfilerCallback())
    trainer = netFoundTrainer(
        label_names=["swappedLabels"],
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=pretraining_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    utils.init_tbwriter(training_args.output_dir)
    trainer.add_callback(utils.StepSyncCallback())
    trainer.add_callback(utils.LearningRateLogCallback(utils.TB_WRITER))
    trainer.add_callback(utils.ThroughputTimingCallback(utils.TB_WRITER))
    utils.start_gpu_logging(training_args.output_dir)
    utils.start_cpu_logging(training_args.output_dir)
    utils.start_ram_logging(training_args.output_dir)

    utils.verify_checkpoint(logger, training_args)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.warning("*** Evaluate ***")
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
