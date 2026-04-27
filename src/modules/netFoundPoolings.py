import torch
import torch.nn as nn

def poolingByConcat(sequence_output, max_burst_length, hidden_size, max_bursts):
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    pads = torch.zeros(
        burstReps.shape[0],
        hidden_size * (max_bursts - burstReps.shape[1]),
        dtype=burstReps.dtype,
        ).to(burstReps.device)
    return torch.concat(
        [torch.reshape(burstReps, (burstReps.shape[0], -1)), pads], dim=-1
    ).to(burstReps.device)


def poolingByMean(sequence_output, attention_mask, max_burst_length):
    burst_attention = attention_mask[:, ::max_burst_length].detach().clone()
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    burst_attention = burst_attention / torch.sum(burst_attention, dim=-1).unsqueeze(
        0
    ).transpose(0, 1)
    orig_shape = burstReps.shape
    burstReps = burst_attention.reshape(
        burst_attention.shape[0] * burst_attention.shape[1], -1
    ) * burstReps.reshape((burstReps.shape[0] * burstReps.shape[1], -1))
    return burstReps.reshape(orig_shape).sum(dim=1)


def poolingByAttention(attentivePooling, sequence_output, max_burst_length):
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    return attentivePooling(burstReps)


class AttentivePooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_dropout = config.hidden_dropout_prob
        self.lin_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)