import inspect
from transformers import Trainer


class netFoundTrainer(Trainer):

    extraFields = {}

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            model_to_inspect = self.model
            if hasattr(self.model, "_orig_mod"):

                model_to_inspect = self.model._orig_mod  # support torch.compiled models
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
        self._signature_columns += {
            "direction",
            "iats",
            "bytes",
            "pkt_count",
            "total_bursts",
            "ports",
            "stats",
            "protocol",
            "dataset_burst_sizes",
        }
        self._signature_columns += self.extraFields
        self._signature_columns = list(set(self._signature_columns))

    def __init__(self, label_names=None, extraFields = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if extraFields is not None:
            self.extraFields = extraFields
        if label_names is not None:
            self.label_names.extend(label_names)
        if kwargs['args'].include_num_input_tokens_seen in {"yes", True, "true", "all"}:
            # fixed in transformers 5.0
            kwargs['args'].include_num_input_tokens_seen = "non_padding"
