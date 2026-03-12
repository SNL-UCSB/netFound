import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import os
from copy import deepcopy
import torch
import torch.distributed
from dataclasses import field, dataclass
from typing import Optional
from torchinfo import summary

from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from torch.distributed.elastic.multiprocessing.errors import record
from datasets.distributed import split_dataset_by_node
from transformers import HfArgumentParser, TrainingArguments

from modules.metrics import classif_metrics, regression_metrics
from modules import utils
from modules.netFoundDataCollator import DataCollatorForFlowClassification
from modules.netFoundModels import netFoundFinetuningModel
from modules.netFoundTrainer import netFoundTrainer
from modules.netFoundTokenizer import netFoundTokenizer

random.seed(42)


@dataclass
class FineTuningDataTrainingArguments(utils.CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    num_labels: int = field(metadata={"help": "number of classes in the datasets"}, default=None)
    problem_type: Optional[str] = field(
        default=None,
        metadata={"help": "Override regression or classification task"},
    )
    p: float = field(
        default=None,
        metadata={
            "help": "noise rate"
        },
    )

def get_label_encoder(problem_type: str, dataset = None, batch_size = 1):
    """
    Labels are strings by default because they are stored in arrow format with predefined str column datatype.
    This function returns mapping function that converts them to class numbers (classification) or float (regression)
    which later is mapped over the finetuning dataset.
    """


    if problem_type == "regression":
        encoder = lambda x: float(x)
        encoder.transform = lambda _b: [float(x) for x in _b]
    else:
        if dataset is None:
            raise ValueError("Dataset should be provided for iteration and getting class names")
        encoder = LabelEncoder()
        dataset = dataset.select_columns("labels")
        labels = set()
        for batch in tqdm(dataset.iter(batch_size=batch_size)):
            labels.update(batch["labels"])
        labels = list(sorted(labels))
        encoder.fit(labels)

    def mapping_function(_batch):
        _batch["labels"] = encoder.transform(_batch["labels"])
        return _batch

    return encoder, mapping_function


@record
def main():
    parser = HfArgumentParser(
        (utils.ModelArguments, FineTuningDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.streaming:
        raise NotImplementedError("Streaming mode for fine-tuning is not implemented.")

    utils.LOGGING_LEVEL = training_args.get_process_log_level()
    logger = utils.get_logger(name=__name__)

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    # Data preparation
    train_dataset, test_dataset = utils.load_train_test_datasets(logger, data_args)

    label_encoder, le_mapping_function = get_label_encoder(data_args.problem_type, train_dataset, batch_size=1024)
    train_dataset = train_dataset.map(function=le_mapping_function, batched=True)
    test_dataset = test_dataset.map(function=le_mapping_function, batched=True)

    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    if "WORLD_SIZE" in os.environ:
        train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]),
                                              world_size=int(os.environ["WORLD_SIZE"]))
        test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]),
                                             world_size=int(os.environ["WORLD_SIZE"]))

    logger.warning("Tokenizing datasets")
    config = utils.update_config(model_args, data_args, training_args, config=None)
    training_tokenizer = netFoundTokenizer(config=config)

    test_config = deepcopy(config)
    test_config.p = 0
    testing_tokenizer = netFoundTokenizer(config=test_config)

    if "WORLD_SIZE" in os.environ and training_args.local_rank > 0:
        logger.warning("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    params = {
        "batched": True,
        "num_proc": data_args.preprocessing_num_workers or utils.get_90_percent_cpu_count(),
    }
    train_dataset = train_dataset.map(function=training_tokenizer, **params)
    test_dataset = test_dataset.map(function=testing_tokenizer, **params)

    if "WORLD_SIZE" in os.environ and training_args.local_rank == 0:
        logger.warning("Loading results from main process")
        torch.distributed.barrier()

    # Model initialization
    labels_dtype = torch.float32 if data_args.problem_type == "regression" else torch.long
    data_collator = DataCollatorForFlowClassification(training_tokenizer.pad_token_id, labels_dtype)
    if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
    ):
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        model = netFoundFinetuningModel.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        model = netFoundFinetuningModel(config=config)
    model = utils.possibly_freeze(model, model_args)
    if os.environ.get("RANK", "0") == "0":
        summary(model)
    if config.compile:
        model = torch.compile(model, mode="max-autotune")

    if data_args.problem_type == "regression":
        compute_metrics = regression_metrics
    else:
        compute_metrics = lambda p: classif_metrics(p, label_encoder)

    training_args.accelerator_config.dispatch_batches = False
    callbacks: list = []
    if data_args.profile:
        callbacks.append(utils.TorchTBProfilerCallback())
    trainer = netFoundTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        tokenizer=testing_tokenizer,
        compute_metrics=compute_metrics,
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
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
