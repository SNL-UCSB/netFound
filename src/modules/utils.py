import dataclasses
import datasets
import transformers
import logging
import os
import json
import threading
import subprocess
import torch
import torch.distributed as dist
import time
import socket
import psutil
import inspect

from collections import defaultdict
from typing import Any, Iterable, Optional, Set, Tuple

from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, schedule, ProfilerActivity
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

LOGGING_LEVEL = logging.WARNING
TB_WRITER: Optional[SummaryWriter] = None
GLOBAL_STEP: int = 0


@dataclasses.dataclass()
class ModelArguments:
    model_name_or_path: str = dataclasses.field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    size: str = dataclasses.field(
        default=None,
        metadata={
            "help": "Model size for config initialization."
        }
    )
    metaFeatures: int = dataclasses.field(
        default=None,
        metadata={"help": "number of metadata fields."},
    )
    num_hidden_layers: int = dataclasses.field(
        default=None,
        metadata={"help": "Number of hidden layers."},
    )
    num_attention_heads: int = dataclasses.field(
        default=None,
        metadata={"help": "Number of attention heads."},
    )
    hidden_size: int = dataclasses.field(
        default=None,
        metadata={"help": "Hidden size."},
    )
    no_ptm: bool = dataclasses.field(
        default=None,
        metadata={"help": "If True, use NoPTM model (only for fine-tuning)."},
    )
    freeze_flow_encoder: bool = dataclasses.field(
        default=None,
        metadata={"help": "Freeze flow encoders"},
    )
    freeze_burst_encoder: bool = dataclasses.field(
        default=None,
        metadata={"help": "Freeze burst encoders"},
    )
    freeze_embeddings: bool = dataclasses.field(
        default=None,
        metadata={"help": "Freeze embeddings"},
    )
    freeze_base: bool = dataclasses.field(
        default=None,
        metadata={"help": "Freeze base model"},
    )
    use_flash_attn: bool = dataclasses.field(
        default=None,
        metadata={"help": "Use flash attention"},
    )
    roformer: bool = dataclasses.field(
        default=None,
        metadata={"help": "Use RoPE"},
    )
    compile: bool = dataclasses.field(
        default=None,
        metadata={"help": "Use torch.compile to compile the model"},
    )
    strip_payload: bool = dataclasses.field(
        default=None,
        metadata={"help": "Strip payload from data"},
    )


@dataclasses.dataclass
class CommonDataTrainingArguments:
    train_dir: Optional[str] = dataclasses.field(
        metadata={"help": "Directory with training data (Apache Arrow files)"})
    test_dir: Optional[str] = dataclasses.field(default=None, metadata={
        "help": "Directory with testing data (Apache Arrow files)"})
    no_meta: bool = dataclasses.field(
        default=None,
        metadata={"help": "no meta fields"},
    )
    flat: bool = dataclasses.field(
        default=None,
        metadata={"help": "no cross burst encoder"},
    )
    limit_bursts: bool = dataclasses.field(
        default=None,
        metadata={"help": "limit_bursts"},
    )
    validation_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Directory with optional input evaluation data to evaluate the perplexity on (Apache Arrow files)"},
    )
    validation_split_percentage: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"}
    )
    data_cache_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "Where to store the dataset cache."},
    )
    overwrite_cache: bool = dataclasses.field(
        default=None,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_bursts: int = dataclasses.field(
        default=None,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_seq_length: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    streaming: bool = dataclasses.field(
        default=None,
        metadata={"help": "Whether to load dataset in the streaming mode."},
    )
    tcpoptions: bool = dataclasses.field(
        default=None,
        metadata={"help": "Whether the data contains TCP options."},
    )
    profile: bool = dataclasses.field(
        default=None,
        metadata={"help": "Use torch profiler to profile the model"},
    )


def update_config(
        model_args: Any,
        data_args: Any,
        training_args: Any,
        config: Optional[Any] = None,
        skip_none: bool = True,
) -> Any:
    """
    Create/update a netFound config by copying matching fields from args.

    Matching rule: copy arg field `x` to `config.x` iff `x` is a known config key
    (either already present on the config instance, or present in config __init__ signature).

    Precedence: model_args < data_args < training_args (later overrides earlier).
    """

    def _args_to_items(obj: Any) -> Iterable[Tuple[str, Any]]:
        if obj is None:
            return []
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                yield f.name, getattr(obj, f.name)
            return
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if not k.startswith("_"):
                    yield k, v
            return
        return []

    if config is None:
        from netFoundConfigs import CONFIG_SIZES
        config = CONFIG_SIZES[model_args.size]()

    _IGNORE_KEYS = {"accelerator_config"}

    for src in (model_args, data_args, training_args):
        for k, v in _args_to_items(src):
            if k in _IGNORE_KEYS:
                continue
            if skip_none and v is None:
                continue
            setattr(config, k, v)

    return config


def possibly_freeze(model, model_args):
    for name, param in model.base_transformer.named_parameters():
        if model_args.freeze_flow_encoder and (
                "flow_encoder" in name or ("encoder" in name and "position_embeddings" in name)):
            param.requires_grad = False
        if model_args.freeze_burst_encoder and "burst_encoder" in name:
            param.requires_grad = False
        if model_args.freeze_embeddings and (name.startswith("embed") or name.startswith("seg_embed")):
            param.requires_grad = False
        if model_args.freeze_base:
            param.requires_grad = False
    return model


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(LOGGING_LEVEL)
    datasets.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


def verify_checkpoint(logger, training_args):
    if not training_args.resume_from_checkpoint:
        folders = set(os.listdir(training_args.output_dir)) - {"runs"}
        if len(folders) > 0 and not training_args.overwrite_output_dir:
            if training_args.local_rank == 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overwrite it."
                )
    else:
        if training_args.local_rank == 0:
            resume_from_checkpoint = training_args.resume_from_checkpoint if isinstance(
                training_args.resume_from_checkpoint, str) else get_last_checkpoint(training_args.output_dir)
            logger.warning(
                f"Checkpoint detected, resuming training at {resume_from_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


def get_90_percent_cpu_count():
    return max(1, int(os.cpu_count() * 0.9))


def load_train_test_datasets(logger, data_args):
    logger.warning("Loading datasets")
    if data_args.test_dir is None:
        data_args.test_dir = data_args.train_dir
        train_split = f"train[{data_args.validation_split_percentage}%:]"
        test_split = f"train[:{data_args.validation_split_percentage}%]"
    else:
        train_split = None
        test_split = None

    train_dataset = load_dataset(
        "arrow",
        data_dir=data_args.train_dir,
        split=train_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )
    if train_split is None:
        train_dataset = train_dataset[list(train_dataset.keys())[0]]

    test_dataset = load_dataset(
        "arrow",
        data_dir=data_args.test_dir,
        split=test_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )
    if test_split is None:
        test_dataset = test_dataset[list(test_dataset.keys())[0]]

    if data_args.max_eval_samples is not None:
        test_dataset = test_dataset.select(
            range(min(test_dataset.shape[0], data_args.max_eval_samples))
        )
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(train_dataset.shape[0], data_args.max_train_samples))
        )

    if not data_args.streaming:
        total_bursts_train = [0] * len(train_dataset)
        total_bursts_test = [0] * len(test_dataset)
    else:
        total_bursts_train = defaultdict(lambda: 0)
        total_bursts_test = defaultdict(lambda: 0)

    train_dataset = train_dataset.add_column("total_bursts", total_bursts_train)
    test_dataset = test_dataset.add_column("total_bursts", total_bursts_test)

    return train_dataset, test_dataset


def init_tbwriter(output_dir=".") -> None:
    global TB_WRITER
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    if not torch.cuda.is_available():
        TB_WRITER = SummaryWriter(os.path.join(output_dir, "runs",
                                               current_time + "_" + socket.gethostname() + f"_pid{os.getpid()}_custom_metrics"))
        return
    TB_WRITER = SummaryWriter(os.path.join(output_dir, "runs",
                                           current_time + "_" + socket.gethostname() + f"_gpu{torch.cuda.current_device()}_custom_metrics"))


def get_gpu_utilization(gpu_id):
    """Fetch GPU utilization using nvidia-smi for the given GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        utilization = int(result.stdout.strip())
        return utilization
    except Exception as e:
        get_logger(__name__).error(f"Error fetching GPU utilization: {e}")
        return 0


def log_gpu_stats(gpu_id, output_dir, interval=10):
    """
    Log GPU utilization and memory usage for the assigned GPU to TensorBoard every `interval` seconds.
    """
    if not torch.cuda.is_available():
        get_logger(__name__).error("No GPU found.")
        return

    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(
        os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_gpu{gpu_id}"))

    while True:
        # Get GPU stats for the current process's assigned GPU
        device = torch.device(f"cuda:{gpu_id}")
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # In GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # In GB
        memory_free = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) - memory_reserved

        # Get GPU utilization using nvidia-smi
        utilization = get_gpu_utilization(gpu_id)

        # Log to TensorBoard
        writer.add_scalar(f"GPU/Memory Allocated (GB)", memory_allocated, time.time())
        writer.add_scalar(f"GPU/Memory Reserved (GB)", memory_reserved, time.time())
        writer.add_scalar(f"GPU/Memory Free (GB)", memory_free, time.time())
        writer.add_scalar(f"GPU/Utilization (%)", utilization, time.time())

        # Sleep before logging the next set of stats
        time.sleep(interval)


def start_gpu_logging(output_dir="."):
    """
    Start logging GPU stats to TensorBoard for the current process's assigned GPU.
    """
    if not torch.cuda.is_available():
        get_logger(__name__).error("No GPU found.")
        return

    gpu_id = torch.cuda.current_device()

    # Start logging GPU stats in a separate thread
    gpu_stats_thread = threading.Thread(target=log_gpu_stats, args=(gpu_id, output_dir))
    gpu_stats_thread.daemon = True
    gpu_stats_thread.start()


def log_ram_stats(output_dir=".", interval=10):
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(
        os.path.join(
            output_dir,
            "runs",
            f"{current_time}_{socket.gethostname()}_ram_metrics"
        )
    )

    while True:
        try:
            mem_info = psutil.virtual_memory()

            # Convert to GB for readability
            writer.add_scalar("RAM/TotalGB", mem_info.total / (1024 ** 3), time.time())
            writer.add_scalar("RAM/UsedGB", mem_info.used / (1024 ** 3), time.time())
            writer.add_scalar("RAM/FreeGB", mem_info.free / (1024 ** 3), time.time())
            writer.add_scalar("RAM/AvailableGB", mem_info.available / (1024 ** 3), time.time())
            writer.add_scalar("RAM/PercentUsed", mem_info.percent, time.time())
            writer.add_scalar("RAM/ActiveGB", mem_info.active / (1024 ** 3), time.time())
            writer.add_scalar("RAM/InactiveGB", mem_info.inactive / (1024 ** 3), time.time())
            writer.add_scalar("RAM/BuffersGB", mem_info.buffers / (1024 ** 3), time.time())
            writer.add_scalar("RAM/CachedGB", mem_info.cached / (1024 ** 3), time.time())
        except Exception as e:
            get_logger(__name__).error(f"Error fetching RAM usage: {e}")
            return 0

        time.sleep(interval)


def start_ram_logging(output_dir="."):
    """
    Start logging overall RAM stats to TensorBoard.
    Only do this in the first local process (SLURM_LOCALID == 0).
    """
    if os.environ.get("SLURM_LOCALID", "-1") != "0":
        return

    ram_stats_thread = threading.Thread(target=log_ram_stats, args=(output_dir,))
    ram_stats_thread.daemon = True
    ram_stats_thread.start()


def log_cpu_stats(output_dir, interval=10):
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(
        os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_cpu_metrics"))

    while True:
        try:
            cpu_load = psutil.cpu_percent(interval=None)
            writer.add_scalar(f"CPU/Utilization %", psutil.cpu_percent(interval=None), time.time())
        except Exception as e:
            get_logger(__name__).error(f"Error fetching CPU utilization: {e}")
            return 0

        time.sleep(interval)


def start_cpu_logging(output_dir="."):
    """
    Start logging overall CPU stats to TensorBoard.
    """
    # do it only for a single process per node
    if os.environ.get("SLURM_LOCALID", "-1") != "0":
        return

    cpu_stats_thread = threading.Thread(target=log_cpu_stats, args=(output_dir,))
    cpu_stats_thread.daemon = True
    cpu_stats_thread.start()


class StepSyncCallback(TrainerCallback):
    """Keep utils.GLOBAL_STEP in sync with the trainer's step counter."""
    def on_step_end(self, args, state, control, **kwargs):
        global GLOBAL_STEP
        GLOBAL_STEP = state.global_step
        return control


class LearningRateLogCallback(TrainerCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def on_step_end(self, args, state, control, **kwargs):
        # The optimizer is passed as a keyword argument
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            # If you have multiple parameter groups, you can log each group’s LR
            for i, param_group in enumerate(optimizer.param_groups):
                self.tb_writer.add_scalar(f"train/learning_rate/group_{i}", param_group['lr'], state.global_step)
        return control

class ThroughputTimingCallback(TrainerCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer
        self._t_step_begin = time.perf_counter()
        self._t_step_end = time.perf_counter()
        self._last_tokens_seen = 0

    def on_step_begin(self, args, state, control, **kwargs):
        cur_time = time.perf_counter()
        batch_wait = cur_time - self._t_step_end
        self.tb_writer.add_scalar("dataloader/batch_wait_in_sec", batch_wait, state.global_step)
        self._t_step_begin = cur_time

        return control

    def on_step_end(self, args, state, control, **kwargs):
        t_end = time.perf_counter()

        step_sec = max(1e-9, t_end - self._t_step_begin)
        self._t_step_end = t_end

        tokens_per_sec = float(state.num_input_tokens_seen - self._last_tokens_seen) / step_sec
        self._last_tokens_seen = state.num_input_tokens_seen
        self.tb_writer.add_scalar("throughput/tokens_per_sec", tokens_per_sec, state.global_step)
        self.tb_writer.add_scalar("throughput/total_tokens_seen", state.num_input_tokens_seen, state.global_step)
        self.tb_writer.add_scalar("throughput/step_duration_in_sec", step_sec, state.global_step)
        return control

class TorchTBProfilerCallback(TrainerCallback):
    def __init__(
        self,
        logdir="tb_profiler",
        skip_first=5,
        wait=5,
        warmup=5,
        active=5,
        repeat=1,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ):
        self.logdir = logdir
        self.prof = None
        self.sched = schedule(
            skip_first=skip_first,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack

    def on_train_begin(self, args, state, control, **kwargs):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        run_dir = os.path.join(args.output_dir, self.logdir, f"rank{rank}")

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.prof = profile(
            activities=activities,
            schedule=self.sched,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=run_dir),
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
        self.prof.start()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.step()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.stop()
        return control