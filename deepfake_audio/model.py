# Vendored from the deepfake_audio_inference drop (2026-07-11). Local change:
# WavLMDeepfakeClassifier accepts a prebuilt WavLMConfig so the checkpoint
# (which carries every weight) can be loaded fully offline instead of
# downloading the HF encoder first.
from __future__ import annotations

import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.attention(states)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1).bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        mean = torch.sum(weights * states, dim=1)
        variance = torch.sum(weights * (states - mean.unsqueeze(1)).pow(2), dim=1)
        return torch.cat([mean, variance.clamp_min(1e-7).sqrt()], dim=-1)


class WavLMDeepfakeClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        dropout: float = 0.2,
        config: WavLMConfig | None = None,
    ):
        super().__init__()
        self.encoder = WavLMModel(config) if config is not None else WavLMModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        layers = self.encoder.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(layers))
        self.pooling = AttentiveStatisticsPooling(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        stacked = torch.stack(outputs.hidden_states, dim=0)
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        states = torch.sum(weights * stacked, dim=0)
        feature_mask = None
        if attention_mask is not None:
            feature_mask = self.encoder._get_feature_vector_attention_mask(states.shape[1], attention_mask)
        return self.classifier(self.pooling(states, feature_mask))

    def set_encoder_trainability(self, last_n_layers: int = 0) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if last_n_layers > 0:
            for layer in self.encoder.encoder.layers[-last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.encoder.encoder.layer_norm.parameters():
                parameter.requires_grad = True
