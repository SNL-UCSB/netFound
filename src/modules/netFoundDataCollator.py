import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from transformers.data import DefaultDataCollator
from typing import Any, Dict, List, Union
import numpy as np
import random
from modules.utils import get_logger

logger = get_logger(name=__name__)

def _pad_batch_field_and_flatten(
        input_data: list[list[int]],
        max_burst_length: int,
        max_bursts: int,
        padding_token: int
) -> np.ndarray:
    # pad each burst
    input_data = [
        burst + [padding_token] * (max_burst_length - len(burst))
        for burst in input_data
    ]

    # pad flow to max_bursts
    input_data += [[padding_token] * max_burst_length] * (max_bursts - len(input_data))

    return np.array(input_data).flatten()


class DataCollatorWithMeta(DataCollatorForLanguageModeling):
    """
    Data collator for pretraining: includes burst swaps, masking metadata, etc.
    """
    def __init__(self, swap_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.swap_rate = swap_rate

    def torch_call(
            self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {}
        bursts_in_each_flow = [example["total_bursts"] for example in examples]
        max_burst_length = max([max(example["dataset_burst_sizes"]) for example in examples]) + 1  # +1 for CLS token
        max_bursts = max(bursts_in_each_flow)
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            for key in inputs.keys():
                if key in {"labels", "total_bursts", "replacedAfter"}:
                    # skip some keys
                    continue
                if key not in batch:
                    # create dummy list for each key
                    batch[key] = []
                if key in {"ports"}:
                    # token id for data is incremented by 1 to reserve 0 for padding
                    batch[key].append(inputs[key] + 1)
                elif key in {"dataset_burst_sizes"}:
                    # pad to max bursts
                    padded_burst_sizes = inputs[key] + [0] * (max_bursts - len(inputs[key]))
                    batch[key].append(padded_burst_sizes)
                elif key in {"protocol", "flow_duration"}:
                    # direct append
                    batch[key].append(inputs[key])
                else:
                    # pad and flatten
                    input_data = _pad_batch_field_and_flatten(inputs[key], max_burst_length=max_burst_length, max_bursts=max_bursts, padding_token=self.tokenizer.pad_token_id)
                    batch[key].append(input_data)
        for key in batch.keys():
            batch[key] = torch.Tensor(np.array(batch[key]))
            if key in {"input_ids", "attention_mask", "ports", "protocol", "dataset_burst_sizes"}:
                batch[key] = batch[key].to(torch.long)

        if self.mlm:
            (
                batch["input_ids"],
                batch["labels"],
                batch["swappedLabels"],
                batch[
                "burstMetasToBeMasked"]
            ) = self.torch_mask_tokens(
                batch["input_ids"],
                bursts_in_each_flow,
                max_burst_length,
                self.swap_rate,
                batch["protocol"],
                special_tokens_mask=None
            )
        else:
            labels = batch["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return BatchEncoding(batch)

    def swap_bursts_adjust_prob_matrix(self, input_ids, bursts_in_each_flow, max_burst_length, swap_rate):
        labels = torch.from_numpy(np.array(np.random.rand(len(bursts_in_each_flow)) < swap_rate, dtype=int))
        swappedIds = []
        for i in range(input_ids.shape[0]):
            if labels[i] == 1:
                burstToRep = random.randint(0, bursts_in_each_flow[i] - 1)
                flowChoice = random.randint(0, input_ids.shape[0] - 1)
                if flowChoice == i:
                    flowChoice = (flowChoice + 1) % input_ids.shape[0]
                burstChoice = random.randint(0, bursts_in_each_flow[flowChoice] - 1)
                swappedIds.append([i, burstToRep])
                input_ids[i][burstToRep * max_burst_length:(burstToRep + 1) * max_burst_length] = input_ids[flowChoice][
                    burstChoice * max_burst_length:(burstChoice + 1) * max_burst_length]
        return input_ids, swappedIds, labels

    def maskMetaData(self, input_ids, bursts_in_each_flow, swapped_bursts):
        masked_meta_bursts = np.full((input_ids.shape[0], max(bursts_in_each_flow)), 0.3)
        for ids in swapped_bursts:
            masked_meta_bursts[ids[0]][ids[1]] = 0
        candidate_flows = np.array(
            [np.array(np.array(bursts_in_each_flow) > 3, dtype=int)]).transpose()  # converting to nX1 matrix
        return torch.bernoulli(torch.from_numpy(candidate_flows * masked_meta_bursts)).bool()

    def torch_mask_tokens(self, input_ids, bursts_in_each_flow, max_burst_length, swap_rate, protos, **kwargs):
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        new_ip_ids, swappedIds, swappedLabels = self.swap_bursts_adjust_prob_matrix(input_ids, bursts_in_each_flow,
                                                                                    max_burst_length, swap_rate)
        mask_metadata = self.maskMetaData(input_ids, bursts_in_each_flow, swappedIds)
        for ids in swappedIds:
            probability_matrix[ids[0]][ids[1] * max_burst_length:(ids[1] + 1) * max_burst_length] = 0
        input_ids = new_ip_ids

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token

        # 10% of the time (50% of remaining 0.2), we replace masked input tokens with random word (1 to max vocab size)
        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        random_words = torch.randint(1, len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels, swappedLabels, mask_metadata


class SimpleDataCollator(DefaultDataCollator):
    """
    Data collator that processes the data and passes labels to output batch

    Args:
        pad_token_id: id of padding token (usually tokenizer.pad_token_id)
        labels_dtype: whether to convert labels to torch.Tensor of a certain dtype or pass as given
    """
    def __init__(self, pad_token_id: int, labels_dtype = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_dtype = labels_dtype
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch = {}
        bursts_in_each_flow = [example["total_bursts"] for example in examples]
        max_burst_length = max([max(example["dataset_burst_sizes"]) for example in examples]) + 1  # +1 for CLS token
        max_bursts = max(bursts_in_each_flow)
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            for key in inputs.keys():
                if key in {"total_bursts", "replacedAfter"}:
                    # skip some keys
                    continue
                if key not in batch:
                    # create dummy list for each key
                    batch[key] = []
                if key in {"ports"}:
                    # token id for data is incremented by 1 to reserve 0 for padding
                    batch[key].append(inputs[key] + 1)
                elif key in {"dataset_burst_sizes"}:
                    # pad to max bursts
                    padded_burst_sizes = inputs[key] + [0] * (max_bursts - len(inputs[key]))
                    batch[key].append(padded_burst_sizes)
                elif key in {"labels", "protocol", "flow_duration"}:
                    # direct append
                    batch[key].append(inputs[key])
                else:
                    # pad and flatten
                    input_data = _pad_batch_field_and_flatten(inputs[key], max_burst_length=max_burst_length, max_bursts=max_bursts, padding_token=self.pad_token_id)
                    batch[key].append(input_data)
        for key in set(batch.keys()) - {"labels"}:
            batch[key] = torch.Tensor(np.array(batch[key]))
            if key in {"input_ids", "attention_mask", "ports", "protocol", "dataset_burst_sizes"}:
                batch[key] = batch[key].to(torch.long)
        if self.labels_dtype is not None:
            batch["labels"] = torch.Tensor(np.array(batch["labels"])).to(self.labels_dtype)
        return BatchEncoding(batch)


class DataCollatorForFlowClassification(SimpleDataCollator):
    """
    Data collator for flow classification

    Args:
        pad_token_id: id of padding token (usually tokenizer.pad_token_id)
        labels_dtype: whether to convert labels to torch.Tensor of a certain dtype or pass as given
    """
    def __init__(self, pad_token_id: int, labels_dtype, *args, **kwargs):
        if labels_dtype is None:
            raise ValueError(
                "labels_dtype cannot be None - labels should be converted to torch.Tensor. "
                "Provide labels_dtype (usually torch.float32 for regression or torch.long for classification)"
            )
        super().__init__(pad_token_id=pad_token_id, labels_dtype=labels_dtype, *args, **kwargs)
