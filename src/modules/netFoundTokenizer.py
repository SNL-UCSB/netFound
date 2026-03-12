import os
import random
from typing import List, Union, Optional, Tuple
import itertools

import numpy as np
from transformers import PreTrainedTokenizer, BatchEncoding
from datasets.formatting.formatting import LazyBatch

BurstType = List[int]
FlowType = List[BurstType]

PROTOCOLS_LENGTH_WITHOUT_PAYLOAD = {
    6: 12,  # TCP
    1: 7,   # ICMP
    17: 6  # UDP
}
PAYLOAD_LENGTH = 6


class netFoundTokenizer(PreTrainedTokenizer):
    bos_token_id = 65537  # aka CLS_TOKEN
    eos_token_id = 65540
    pad_token_id = 0
    mask_token = 65538
    vocab_size = 65537  # 0-65536 inclusive
    ATTN_PRESENCE_TOKEN = 1
    ATTN_ABSENCE_TOKEN = 0

    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.max_bursts = config.max_bursts
        self.max_burst_length = config.max_burst_length
        self.p = config.p
        self.pretraining = config.pretraining
        self.name_or_path = config.name_or_path
        self.limit_bursts = config.limit_bursts
        self.raw_labels = False
        self.strip_payload = config.strip_payload

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, max_bursts={self.max_bursts}, max_burst_length={self.max_burst_length}, p={self.p})"
        )

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return [self.bos_token_id, self.pad_token_id]

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
    ) -> Tuple[str]:
        return

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def truncate_flow(flow: FlowType, max_bursts: int, max_burst_length: int) -> FlowType:
        """
        Truncate the flow to `max_bursts` and each burst to `max_burst_length`.
        """
        return [burst[:max_burst_length] for burst in flow][:max_bursts]

    @staticmethod
    def prepend_to_list(flow: FlowType, token: Optional[int]) -> FlowType:
        # Sometimes we prepend CLS_TOKEN (aka bos_token_id)
        if token is not None:
            return [[token] + burst for burst in flow]
        else:
            # for metadata - just repeat the first value for CLS_TOKEN position
            return [[burst[0]] + burst for burst in flow]

    @staticmethod
    def convert_to_tokens(flow: FlowType, add_one: bool = False) -> FlowType:
        if not add_one:
            return flow  # noop
        return [[tok + add_one for tok in burst] for burst in flow]

    @staticmethod
    def convert_to_attn(bursts: FlowType) -> FlowType:
        return [[1] * len(burst) for burst in bursts]

    def __call__(self, dataset):
        return self.tokenize(dataset)

    @staticmethod
    def _expand_bursts(flows: list[list[int]], burst_sizes: list[list[int]]) -> list[list[list[int]]]:
        """
        To save space, some repetitive info is stored as a single value for the entire burst.
        This function expands the burst sizes to match the actual burst lengths.
        """
        return [
            [
                [value] * burst_sizes[idx][i]
                for i, value in enumerate(flow)
            ]
            for idx, flow in enumerate(flows)
        ]

    @staticmethod
    def multiply_burst_values(flows: list[list[float]], multiplier: float, ftype=float) -> list[list[float]]:
        return [
            [ftype(burst_value * multiplier) for burst_value in flow]
            for flow in flows
        ]

    @staticmethod
    def _strip_payload(burst_tokens: List[FlowType], protocols: List[int]) -> List[FlowType]:
        """
        `burst_tokens` is List[flow][burst][token].
        Dataset tokens include payload; for payload-disabled runs we keep only header tokens.
        """
        stripped_burst_tokens = []
        for flow_idx, flow in enumerate(burst_tokens):
            proto = protocols[flow_idx]
            header_length = PROTOCOLS_LENGTH_WITHOUT_PAYLOAD.get(proto)
            if header_length is None:
                raise ValueError(f"Unsupported protocol {proto} for payload stripping.")

            stripped_flow: List[List[int]] = []
            for burst in flow:
                new_burst = list(itertools.chain.from_iterable(
                    burst[i: i + header_length]
                    for i in range(0, len(burst), header_length + PAYLOAD_LENGTH)  # header + payload in current dataset
                ))
                stripped_flow.append(new_burst)
            stripped_burst_tokens.append(stripped_flow)
        return stripped_burst_tokens

    def tokenize_fields(
            self,
            dataset: list[FlowType],
            prepend_token: int = None,
            add_one: bool = False
    ) -> list[FlowType]:
        return [
            self.truncate_flow(
                self.prepend_to_list(self.convert_to_tokens(flow, add_one), prepend_token),
                max_bursts=self.max_bursts,
                max_burst_length=self.max_burst_length,
            )
            for flow in dataset
        ]


    def tokenize_fields_with_attn(
            self,
            dataset: list[FlowType],
            prepend_token: int = None,
            add_one: bool = False
    ) -> tuple[list[FlowType], list[FlowType]]:
        tokenized_data = self.tokenize_fields(dataset, prepend_token, add_one)
        attn = [
            self.truncate_flow(
                self.prepend_to_list(self.convert_to_attn(flow), self.ATTN_PRESENCE_TOKEN),
                max_burst_length=self.max_burst_length,
                max_bursts=self.max_bursts,
            )
            for flow in dataset
        ]
        return tokenized_data, attn

    def tokenize(self, text, **kwargs):
        dataset: LazyBatch = text
        dataset['iats'] = self.multiply_burst_values(dataset['iats'], 1e-3, int)

        if self.strip_payload:
            dataset['burst_tokens'] = self._strip_payload(dataset['burst_tokens'], dataset['protocol'])
        dataset_burst_sizes = [[len(burst) for burst in flow] for flow in dataset["burst_tokens"]]

        if not self.pretraining and "labels" in dataset:
            labels = np.array(dataset["labels"], dtype=int)
            if self.p > 0:
                num_noise_samples = int(self.p * len(labels))
                indices = random.sample(range(0, len(labels) - 1), num_noise_samples)
                noisy_labels = np.random.random_integers(
                    labels.min(), labels.max(), size=(num_noise_samples,)
                )
                labels[indices] = noisy_labels
            labels = labels.tolist()

        # restore directions: true/false -> 1/-1
        direction = [[1 if direction else -1 for direction in flow] for flow in dataset["directions"]]
        direction = self.tokenize_fields(self._expand_bursts(direction, dataset_burst_sizes))

        pkt_bytes = self.tokenize_fields(self._expand_bursts(dataset["bytes"], dataset_burst_sizes))
        pkt_count = self.tokenize_fields(self._expand_bursts(dataset["counts"], dataset_burst_sizes))
        iats = self.tokenize_fields(self._expand_bursts(dataset["iats"], dataset_burst_sizes))
        input_ids, attention_mask = self.tokenize_fields_with_attn(
            dataset["burst_tokens"], prepend_token=self.bos_token_id, add_one=True
        )
        total_bursts = [len(flow) for flow in dataset["burst_tokens"]]

        batchDict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "direction": direction,
            "bytes": pkt_bytes,
            "pkt_count": pkt_count,
            "iats": iats,
            "total_bursts": total_bursts,
            "flow_duration": dataset["flow_duration"],
            "protocol": dataset["protocol"],
            "dataset_burst_sizes": dataset_burst_sizes,
        }
        if not self.pretraining and "labels" in dataset:
            batchDict.update({"labels": labels})
        if self.raw_labels:
            batchDict.update({"labels": dataset["labels"]})

        return BatchEncoding(batchDict)
