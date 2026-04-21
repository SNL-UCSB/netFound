"""
netFound inference script for extracting embeddings from an Arrow dataset.

This script loads a pretrained netFound checkpoint, encodes every example in
the input dataset, and saves a pickle file containing `(embeddings, labels)`.

Example:
    python src/netFoundInference.py \
        --model_name_or_path /path/to/checkpoint \
        --size small \
        --train_dir /path/to/arrow/dataset \
        --output_file /path/to/embeddings.pkl \
        --output_dir /tmp/netfound_inference \
        --per_device_eval_batch_size 64 \
        --dataloader_num_workers 8 \
        --bf16
"""

import pickle
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments

from modules import utils
from modules.netFoundDataCollator import SimpleDataCollator
from modules.netFoundModels import netFoundLanguageModelling
from modules.netFoundTokenizer import netFoundTokenizer


@dataclass
class InferenceArguments:
    output_file: str = field(
        metadata={"help": "Path to the output pickle file for (embeddings, labels)."}
    )
    overwrite_output_file: bool = field(
        default=False,
        metadata={"help": "Overwrite output_file if it already exists."},
    )


def encode_batch(batch, model):
    device = next(model.base_transformer.parameters()).device
    dtype = next(model.base_transformer.parameters()).dtype

    with torch.inference_mode():
        batch_size, seq_len = batch["input_ids"].shape
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        outputs = model.base_transformer(
            input_ids=batch["input_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            position_ids=position_ids,
            direction=batch["direction"].to(device, dtype=dtype, non_blocking=True),
            iats=batch["iats"].to(device, dtype=dtype, non_blocking=True),
            bytes=batch["bytes"].to(device, dtype=dtype, non_blocking=True),
            pkt_count=batch["pkt_count"].to(device, dtype=dtype, non_blocking=True),
            protocol=batch["protocol"].to(device, non_blocking=True),
            dataset_burst_sizes=batch["dataset_burst_sizes"].to(device, non_blocking=True),
            return_dict=True,
        ).last_hidden_state
        embeddings = torch.mean(outputs, dim=1).float().cpu()

    labels = batch.get("labels")
    if labels is None:
        return embeddings, None
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    else:
        labels = list(labels)
    return embeddings, labels


def _encode_batch_worker(slot, batch, model, results):
    results[slot] = encode_batch(batch, model)


def get_embeddings(
    data_loader,
    models,
):
    if not models:
        raise ValueError("No models were provided for inference.")

    result_embeddings = []
    result_labels = None
    iterator = iter(data_loader)
    round_idx = 0
    total_rounds = (len(data_loader) + len(models) - 1) // len(models)

    progress = tqdm(total=total_rounds, desc="Extracting embeddings")
    while True:
        batches = []
        for _ in range(len(models)):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        if not batches:
            break

        results = [None] * len(batches)
        threads = []
        for idx, batch in enumerate(batches):
            worker = threading.Thread(
                target=_encode_batch_worker,
                args=(idx, batch, models[idx], results),
            )
            worker.start()
            threads.append(worker)

        for worker in threads:
            worker.join()

        for embeddings, labels in results:
            result_embeddings.append(embeddings)
            if labels is not None:
                if result_labels is None:
                    result_labels = []
                result_labels.extend(labels)

        round_idx += 1
        progress.update(1)
        if round_idx % 50 == 0:
            torch.cuda.empty_cache()

    progress.close()

    all_embeddings = torch.cat(result_embeddings, dim=0)
    return all_embeddings, result_labels


@record
def main():
    parser = HfArgumentParser(
        (utils.ModelArguments, utils.CommonDataTrainingArguments, TrainingArguments, InferenceArguments)
    )
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    utils.LOGGING_LEVEL = training_args.get_process_log_level()
    logger = utils.get_logger(name=__name__)

    if data_args.streaming:
        raise NotImplementedError("Streaming mode for inference is not implemented.")
    if data_args.max_train_samples is not None:
        data_args.max_train_samples = int(data_args.max_train_samples)

    model_path = model_args.model_name_or_path
    output_path = Path(inference_args.output_file)

    if model_path is None:
        raise ValueError("Provide --model_name_or_path for inference.")
    if not Path(data_args.train_dir).exists():
        raise FileNotFoundError(f"Data path does not exist: {data_args.train_dir}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {model_path}")
    if output_path.exists() and not inference_args.overwrite_output_file:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite_output_file to replace it."
        )

    logger.warning(f"model_args: {model_args}")
    logger.warning(f"data_args: {data_args}")
    logger.warning(f"training_args: {training_args}")
    logger.warning(f"inference_args: {inference_args}")

    config = utils.update_config(model_args, data_args, training_args, config=None)
    config.pretraining = False
    dtype = torch.bfloat16 if (training_args.bf16 or training_args.bf16_full_eval) else torch.float32

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs are available. netFoundInference requires CUDA.")
    logger.warning(
        f"Using {num_gpus} visible GPU(s) with dtype="
        f"{dtype}."
    )

    models = []
    for gpu_idx in range(num_gpus):
        model = netFoundLanguageModelling.from_pretrained(
            model_path,
            config=config,
        )
        model.to(device=f"cuda:{gpu_idx}", dtype=dtype)
        if config.compile:
            model.base_transformer = torch.compile(
                model.base_transformer,
                mode="max-autotune",
            )
        model.eval()
        models.append(model)
    torch.cuda.empty_cache()

    load_args = replace(data_args, test_dir=data_args.train_dir)
    dataset, _ = utils.load_train_test_datasets(logger, load_args)
    if len(dataset) == 0:
        raise ValueError("Inference dataset is empty after applying max_train_samples.")

    tokenizer = netFoundTokenizer(config=config)
    tokenizer.pretraining = True
    tokenizer.raw_labels = "labels" in dataset.column_names

    map_kwargs = {
        "function": tokenizer,
        "batched": True,
    }
    if not data_args.streaming:
        map_kwargs["num_proc"] = (
            data_args.preprocessing_num_workers or utils.get_90_percent_cpu_count()
        )
        if data_args.overwrite_cache is not None:
            map_kwargs["load_from_cache_file"] = not data_args.overwrite_cache

    logger.warning("Tokenizing inference dataset")
    dataset = dataset.map(**map_kwargs)
    columns_to_remove = [
        column
        for column in ("burst_tokens", "directions", "counts")
        if column in dataset.column_names
    ]
    dataset = dataset.remove_columns(columns_to_remove)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": training_args.per_device_eval_batch_size,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": SimpleDataCollator(pad_token_id=tokenizer.pad_token_id),
    }
    if training_args.dataloader_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    data_loader = torch.utils.data.DataLoader(**loader_kwargs)

    embeddings, labels = get_embeddings(
        data_loader,
        models,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump((embeddings, labels), handle)

    logger.warning(f"Saved embeddings with shape {tuple(embeddings.shape)} to {output_path}")


if __name__ == "__main__":
    main()
