
import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

from modules.netFoundEmbeddings import netFoundRoformerEmbeddings, netFoundRobertaEmbeddings
from modules.netFoundLayers import netFoundEncoder, BaseModelOutputWithFlowAttentions, LMHead
from modules.netFoundPoolings import AttentivePooling, poolingByAttention 

import modules.utils as utils

logger = utils.get_logger(__name__)

TORCH_IGNORE_INDEX = -100


class netFoundPretrainedModel(PreTrainedModel):
    """
    A base pretrained class for netFound models
    """

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [
                k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore
            ]
            self._keys_to_ignore_on_load_missing = [
                k
                for k in self._keys_to_ignore_on_load_missing
                if k not in del_keys_to_ignore
            ]

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)


class netFoundBaseModel(netFoundPretrainedModel):
    """
    Base model containing encoder
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.roformer:
            self.embeddings = netFoundRoformerEmbeddings(config)
        else:
            self.embeddings = netFoundRobertaEmbeddings(config)
        self.encoder = netFoundEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            direction=None,
            iats=None,
            bytes=None,
            pkt_count=None,
            protocol=None,
            dataset_burst_sizes=None,
    ):
        batch_max_burst_length = dataset_burst_sizes.max().item() + 1  # +1 for CLS token
        embeddings = self.embeddings(
            input_ids, position_ids, direction, iats, bytes, pkt_count, protocol
        )
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        num_bursts = input_ids.shape[-1] // batch_max_burst_length
        encoder_outputs = self.encoder(
            hidden_states=embeddings,
            attention_mask=extended_attention_mask,
            num_bursts=num_bursts,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            batch_max_burst_length=batch_max_burst_length
        )
        final_output = encoder_outputs[0]

        if not return_dict:
            return (final_output) + encoder_outputs[1:]

        return BaseModelOutputWithFlowAttentions(
            last_hidden_state=final_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            flow_attentions=encoder_outputs.flow_attentions,
        )


class netFoundLanguageModelling(netFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _tied_weights_keys = [
        "base_transformer.encoder.layer.*.position_embeddings.weight",
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.base_transformer = netFoundBaseModel(config)
        self.lm_head = LMHead(config)
        self.no_mlm = config.no_mlm
        self.no_swapped_bursts = config.no_swapped_bursts
        self.no_metadata_loss = config.no_metadata_loss
        self.no_direction_loss = config.no_direction_loss
        self.mlm_loss_weight = getattr(config, 'mlm_loss_weight', 1.0)
        self.swap_loss_weight = getattr(config, 'swap_loss_weight', 1.0)
        self.metadata_loss_weight = getattr(config, 'metadata_loss_weight', 1.0)
        self.direction_loss_weight = getattr(config, 'direction_loss_weight', 1.0)
        self.attentivePooling = AttentivePooling(config)
        self.swappedClassifierHiddenLayer = nn.Linear(config.hidden_size, 2)
        self.linearMetadataPred = nn.Linear(config.hidden_size, 3)
        self.dirPred = nn.Linear(config.hidden_size, 2)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.base_transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_transformer.set_input_embeddings(value)

    @staticmethod
    def maskMeta(bursts_to_mask, metaFeature, max_burst_length):
        # Expand burst-level mask to token-level and zero masked spans at once
        burst_mask = bursts_to_mask.to(torch.bool).unsqueeze(-1)
        token_mask = burst_mask.expand(-1, -1, max_burst_length).reshape(
            bursts_to_mask.shape[0], -1
        )
        return metaFeature.masked_fill(token_mask, 0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            direction=None,
            iats=None,
            bytes=None,
            pkt_count=None,
            ports=None,
            swappedLabels=None,
            burstMetasToBeMasked=None,
            protocol=None,
            dataset_burst_sizes=None,
            **kwargs
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_max_burst_length = dataset_burst_sizes.max().item() + 1  # +1 for CLS token

        # creating ground truths tensors before masking
        direction_orig = direction.clone().to(torch.long)
        iat_orig = iats.clone() / 1000  # adjusting as values are higher.
        bytes_orig = bytes.clone() / 1000  # adjusting as values are higher.
        pktCount_orig = pkt_count.clone()

        direction = self.maskMeta(burstMetasToBeMasked, direction, batch_max_burst_length)
        iats = self.maskMeta(burstMetasToBeMasked, iats, batch_max_burst_length)
        bytes = self.maskMeta(burstMetasToBeMasked, bytes, batch_max_burst_length)
        pktCount = self.maskMeta(burstMetasToBeMasked, pkt_count, batch_max_burst_length)
        outputs = self.base_transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
            iats=iats,
            bytes=bytes,
            pkt_count=pktCount,
            protocol=protocol,
            dataset_burst_sizes=dataset_burst_sizes,
        )

        # mlm prediction
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # swapped bursts predictions
        pooled_output = poolingByAttention(
            self.attentivePooling, sequence_output, batch_max_burst_length
        )
        swappedLogits = self.swappedClassifierHiddenLayer(pooled_output)

        # metadata prediction except direction
        burstReps = sequence_output[:, ::batch_max_burst_length, :]
        burstMetaFieldsToBeMasked = burstMetasToBeMasked.unsqueeze(dim=2).expand(-1, -1,
                                                                                 self.linearMetadataPred.bias.shape[
                                                                                     -1]).to(torch.float32)
        metaPreds = self.linearMetadataPred(burstReps) * burstMetaFieldsToBeMasked
        metaLabels = burstMetaFieldsToBeMasked * torch.stack([
            iat_orig[:, ::batch_max_burst_length],
            bytes_orig[:, ::batch_max_burst_length],
            pktCount_orig[:, ::batch_max_burst_length]
        ], dim=2)

        # metadata prediction - direction
        # direction will be a classification task, -100 is used to not compute loss in pytorch.
        # All the unmasked values will be set to 0, so we remove the 0 directions.
        direction_orig_ = direction_orig[:, ::batch_max_burst_length]
        direction_orig_ = burstMetasToBeMasked.to(torch.long) * direction_orig_
        direction_orig_[direction_orig_.to(torch.long) == 0] = TORCH_IGNORE_INDEX
        # We have +1 -1 as direction, but for classification we need 0 1. Setting -1 as 0 for classification
        direction_orig_[direction_orig_.to(torch.long) == -1] = 0
        direction_logits = self.dirPred(burstReps)

        losses = []
        ce_loss = nn.CrossEntropyLoss(ignore_index=TORCH_IGNORE_INDEX)
        prefix = "train" if self.training else "eval"
        global_step = utils.GLOBAL_STEP
        if not self.no_mlm:
            masked_lm_loss = ce_loss(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            losses.append(self.mlm_loss_weight * masked_lm_loss)
            utils.TB_WRITER.add_scalar(
                tag=f"{prefix}/mlm_loss",
                scalar_value=masked_lm_loss.item(),
                global_step=global_step,
            )

        if not self.no_swapped_bursts:
            swappedClassificationLoss = ce_loss(swappedLogits, swappedLabels)
            losses.append(self.swap_loss_weight * swappedClassificationLoss)
            utils.TB_WRITER.add_scalar(
                tag=f"{prefix}/swap_bursts_loss",
                scalar_value=swappedClassificationLoss.item(),
                global_step=global_step,
            )

        if not self.no_metadata_loss:
            # Compute L1 only over masked positions: default reduction='mean' would
            # divide by ALL elements (including unmasked zeros), diluting the signal.
            num_masked_elements = burstMetaFieldsToBeMasked.sum().clamp(min=1)
            metaLoss = nn.functional.l1_loss(
                metaPreds, metaLabels.to(metaPreds.dtype), reduction='sum'
            ) / num_masked_elements
            losses.append(self.metadata_loss_weight * metaLoss)
            utils.TB_WRITER.add_scalar(
                tag=f"{prefix}/metadata_loss",
                scalar_value=metaLoss.item(),
                global_step=global_step,
            )

        # transpose for k-dimension loss that wants (BATCH x CLASS_NUMBER x OTHER_DIMENSION)
        if not self.no_direction_loss:
            if (direction_orig_ != TORCH_IGNORE_INDEX).any():
                dirLoss = ce_loss(direction_logits.transpose(1, 2), direction_orig_)
            else:
                # if all labels are -100 - loss is nan: https://github.com/pytorch/pytorch/issues/70348 - let's do like facebook: https://github.com/facebookresearch/detectron2/commit/04fc85a0c44675559c2fbc9c7541cbb8b443819c
                dirLoss = direction_logits.sum() * 0

            if not torch.isnan(dirLoss):
                losses.append(self.direction_loss_weight * dirLoss)
            utils.TB_WRITER.add_scalar(
                tag=f"{prefix}/direction_loss",
                scalar_value=dirLoss.item(),
                global_step=global_step,
            )

        if not losses:
            raise ValueError("No valid losses are defined")

        totalLoss = torch.stack(losses).sum()
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (totalLoss,) + output

        return MaskedLMOutput(
            loss=totalLoss,
            logits=(prediction_scores, swappedLogits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def guess_problem_type(labels, num_labels):
    if num_labels == 1:
        return "regression"
    elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        return "single_label_classification"
    else:
        return "multi_label_classification"


def calculate_finetuning_loss(logits, labels, problem_type, num_labels):
    loss = None
    match problem_type:
        case "regression":
            loss_fct = nn.L1Loss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), (labels.squeeze().to(torch.float32)))
            else:
                loss = loss_fct(logits, labels)
        case "single_label_classification":
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels)
        case "multi_label_classification":
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        case _:
            raise ValueError(f"Unknown problem type: {problem_type}")
    return loss


class netFoundFinetuningModel(netFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _tied_weights_keys = ["base_transformer.encoder.layer.*.position_embeddings.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model_max_length = config.model_max_length
        self.base_transformer = netFoundBaseModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.mlp_dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation = nn.GELU()
        self.hiddenLayer = nn.Linear(config.hidden_size, config.hidden_size)
        self.hiddenLayer2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.attentivePooling = AttentivePooling(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            direction=None,
            iats=None,
            bytes=None,
            pkt_count=None,
            protocol=None,
            stats=None,
            flow_duration=None,
            dataset_burst_sizes=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if self.config.problem_type is None:
            self.config.problem_type = guess_problem_type(labels, self.num_labels)
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        batch_max_burst_length = dataset_burst_sizes.max().item() + 1  # +1 for CLS token

        outputs = self.base_transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
            iats=iats,
            bytes=bytes,
            pkt_count=pkt_count,
            protocol=protocol,
            dataset_burst_sizes=dataset_burst_sizes,
        )

        sequence_output = outputs[0]
        pooled_output = poolingByAttention(
            self.attentivePooling, sequence_output, batch_max_burst_length
        )
        pooled_output = self.hiddenLayer2(self.dropout(self.activation(self.hiddenLayer(pooled_output))))
        logits = self.classifier(pooled_output)
        loss = calculate_finetuning_loss(logits, labels, self.config.problem_type, self.num_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class netFoundNoPTM(netFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model_max_length = config.model_max_length
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.mlp_dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.hiddenLayer = nn.Linear(1595, config.hidden_size * 2)
        self.hiddenLayer2 = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            direction=None,
            iat=None,
            bytes=None,
            pkt_count=None,
            protocol=None,
            stats=None,
            flow_duration=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if self.config.problem_type is None:
            self.config.problem_type = guess_problem_type(labels, self.num_labels)
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input = torch.concatenate(
            [
                input_ids,
                torch.zeros((input_ids.shape[0], 1595 - input_ids.shape[1])).to(
                    input_ids.device
                ),
            ],
            dim=-1,
        )

        pooled_output = self.hiddenLayer2(self.hiddenLayer(input))
        logits = self.classifier(torch.concatenate([pooled_output], dim=-1))
        loss = calculate_finetuning_loss(logits, labels, self.config.problem_type, self.num_labels)

        if not return_dict:
            output = (logits,) + pooled_output[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
