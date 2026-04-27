from transformers import PretrainedConfig

from modules.netFoundTokenizer import netFoundTokenizer


class netFoundConfig(PretrainedConfig):
    """
    Base configuration class for netFound model with payload
    """
    model_type = "netFound"

    # embedding size in our model should always be equal to hidden size so we tie them together
    attribute_map = {
        "embedding_size": "hidden_size",
    }

    def _assign_attrs(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __init__(
            self,
            vocab_size=65600,
            hidden_size=768,
            max_bursts=12,
            max_burst_length=108 + 1,
            model_max_length=1296 + 12,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=1152,
            hidden_activation="gelu_new",
            hidden_dropout_prob=0.1,
            mlp_dropout=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=108 + 1,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            encoder_layout=None,
            use_cache=True,
            classifier_dropout=None,
            metaFeatures=4,
            roformer=True,
            no_meta=False,
            flat=False,
            no_mlm=False,
            no_swapped_bursts=False,
            rep_output_path=None,
            subflow_bursts=3,
            no_metadata_loss=False,
            no_direction_loss=False,
            use_flash_attn=False,
            mlp_bias=False,
            pretraining=True,
            p=0.0,
            limit_bursts=False,
            rotary_value=False,
            subflow_len=-1,
            norm_eps=1e-5,
            norm_bias=False,
            mlm_probability=0.3,
            swap_rate=0.5,
            compile=False,
            strip_payload=False,
            mlm_loss_weight=1.0,
            swap_loss_weight=1.0,
            metadata_loss_weight=1.0,
            direction_loss_weight=1.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._assign_attrs(**params)

        self.eos_token_id = netFoundTokenizer.eos_token_id
        self.bos_token_id = netFoundTokenizer.bos_token_id
        self.pad_token_id = netFoundTokenizer.pad_token_id


class netFoundTCPOptionsConfig(netFoundConfig):
    """Base configuration class for netFound model with TCP options added."""

    def __init__(self, *args, **kwargs):
        # 6 packets max * (18 tokens max for tcp + 20 tokens for tcpoptions) + 1 for CLS
        max_burst_length = 6 * (18 + 20) + 1
        kwargs["max_burst_length"] = max_burst_length
        kwargs["max_position_embeddings"] = max_burst_length

        # (6 packets * (18 tokens + 20 tcpoptions) + 1 CLS token) * 12 bursts
        kwargs["model_max_length"] = max_burst_length * 12
        super().__init__(*args, **kwargs)


class netFoundNoPayloadConfig(netFoundConfig):
    """Base configuration class for netFound model without payload processing."""

    def __init__(self, *args, **kwargs):
        # 6 packets max * 12 tokens max for tcp without payload + 1 for CLS
        max_burst_length = 6 * 12 + 1
        kwargs["max_burst_length"] = max_burst_length
        kwargs["max_position_embeddings"] = max_burst_length

        # (6 packets * 12 tokens + 1 CLS token) * 12 bursts
        kwargs["model_max_length"] = max_burst_length * 12

        kwargs["strip_payload"] = True
        super().__init__(*args, **kwargs)
