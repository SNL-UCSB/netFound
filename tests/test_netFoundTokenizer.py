import pytest

from modules.netFoundTokenizer import netFoundTokenizer
from modules.netFoundConfigBase import netFoundConfig


@pytest.fixture
def tokenizer():
    # Use a tiny config to keep shapes small and predictable
    cfg = netFoundConfig(
        vocab_size=20,
        max_bursts=2,
        max_burst_length=4,
        name_or_path="dummy-tokenizer",
    )
    return netFoundTokenizer(cfg)

def test_expand_bursts(tokenizer):
    flows = [[1, 2, 3, 4], [5]]
    expanded = tokenizer._expand_bursts(flows, [[2, 2, 2, 2], [3]])
    expected = [[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5, 5]]]
    assert expanded == expected


def test_tokenize_builds_padded_fields(tokenizer):
    dataset = {
        "burst_tokens": [
            [[0, 1], [2]],         # flow 1: 2 bursts (lengths 2 and 1)
            [[5, 6, 7]],           # flow 2: 1 burst (length 3)
        ],
        "iats": [
            [1000, 2000],
            [3000],
        ],
        "bytes": [
            [10, 20],
            [30],
        ],
        "counts": [
            [2, 1],
            [3],
        ],
        "directions": [
            [True, False],
            [True],
        ],
        "flow_duration": [1.5, 2.5],
        "protocol": [6, 17],
    }

    batch = tokenizer.tokenize(dataset)

    # Expected shapes will be dynamic because padding is done by collator on batch level
    # actual values are vals+1 because we add one to values to correctly tokenize zeroes as a valid token
    expected_dataset_burst_sizes = [[2, 1], [3]]
    expected_input_ids = [
        [[tokenizer.bos_token_id, 1, 2], [tokenizer.bos_token_id, 3]],
        [[tokenizer.bos_token_id, 6, 7, 8,]]
    ]
    expected_attention = [
        [[1, 1, 1], [1, 1]],
        [[1, 1, 1, 1]]
    ]
    expected_direction = [
        [[1, 1, 1], [-1, -1]],
        [[1, 1, 1, 1]]
    ]
    expected_bytes = [
        [[10, 10, 10], [20, 20]],
        [[30, 30, 30, 30]]
    ]
    expected_counts = [
        [[2, 2, 2], [1, 1]],
        [[3, 3, 3, 3]]
    ]
    expected_iats = [
        [[1, 1, 1], [2, 2]],
        [[3, 3, 3, 3]]
    ]

    assert batch["dataset_burst_sizes"] == expected_dataset_burst_sizes
    assert batch["input_ids"] == expected_input_ids
    assert batch["attention_mask"] == expected_attention
    assert batch["direction"] == expected_direction
    assert batch["bytes"] == expected_bytes
    assert batch["pkt_count"] == expected_counts
    assert batch["iats"] == expected_iats
    assert batch["total_bursts"] == [2, 1]
    assert batch["flow_duration"] == dataset["flow_duration"]
    assert batch["protocol"] == dataset["protocol"]
