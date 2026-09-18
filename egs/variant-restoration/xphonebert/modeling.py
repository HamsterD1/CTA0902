"""Small trainable modules for frozen-Qwen XPhoneBERT fusion."""

from __future__ import annotations

import torch
from torch import nn


class PhonemeResampler(nn.Module):
    """Cross-attend Qwen token embeddings to frozen XPhoneBERT states."""

    def __init__(self, qwen_hidden_size: int = 5120, phoneme_hidden_size: int = 768) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(qwen_hidden_size)
        self.query_projection = nn.Linear(qwen_hidden_size, phoneme_hidden_size, bias=False)
        self.attention = nn.MultiheadAttention(phoneme_hidden_size, num_heads=12, dropout=0.0, batch_first=True)

    def forward(self, text_embeddings: torch.Tensor, phoneme_states: torch.Tensor, phoneme_attention_mask: torch.Tensor) -> torch.Tensor:
        queries = self.query_projection(self.query_norm(text_embeddings))
        return self.attention(
            query=queries,
            key=phoneme_states,
            value=phoneme_states,
            key_padding_mask=~phoneme_attention_mask.bool(),
            need_weights=False,
        )[0]


class PhoneticProjector(nn.Module):
    def __init__(self, input_size: int = 768, output_size: int = 5120) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 2048), nn.GELU(), nn.Linear(2048, output_size))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class ResidualFusion(nn.Module):
    """A bounded scalar starts at exactly the text-only model."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.tanh(self.alpha_raw)

    def forward(self, text_embeddings: torch.Tensor, phonetic_embeddings: torch.Tensor, variant_mask: torch.Tensor) -> torch.Tensor:
        return text_embeddings + self.alpha * phonetic_embeddings * variant_mask.unsqueeze(-1).to(phonetic_embeddings.dtype)


class ExplicitIpaAdapter(nn.Module):
    """Train only the newly-added atomic IPA token embeddings, not Qwen."""

    def __init__(self, special_token_ids: list[int], hidden_size: int = 5120) -> None:
        super().__init__()
        self.register_buffer("special_token_ids", torch.tensor(special_token_ids, dtype=torch.long), persistent=True)
        self.embeddings = nn.Embedding(len(special_token_ids), hidden_size)

    def apply(self, input_ids: torch.Tensor, input_embedding: nn.Module, fallback_token_id: int) -> torch.Tensor:
        """Replace special token ids before base lookup and overlay standalone embeddings."""
        positions = input_ids.new_full(input_ids.shape, -1)
        for index, token_id in enumerate(self.special_token_ids.tolist()):
            positions.masked_fill_(input_ids == token_id, index)
        selected = positions >= 0
        safe_ids = input_ids.masked_fill(selected, fallback_token_id)
        result = input_embedding(safe_ids).clone()
        if selected.any():
            result[selected] = self.embeddings(positions[selected])
        return result
