"""Frozen-Qwen adapter wrappers with compact checkpoint state dicts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from modeling import ExplicitIpaAdapter, PhonemeResampler, PhoneticProjector, ResidualFusion
from model_contract import text_hidden_size


def freeze(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


class AdapterBase(nn.Module):
    def __init__(self, qwen: nn.Module, condition: str):
        super().__init__()
        self.qwen = qwen
        self.condition = condition
        self.config = qwen.config
        freeze(qwen)

    def train(self, mode: bool = True):
        """Keep the frozen language model deterministic when Trainer enables training."""
        super().train(mode)
        self.qwen.eval()
        return self

    def save_pretrained(self, directory: str | Path, **_: object) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "adapter.pt")
        (directory / "adapter_config.json").write_text(json.dumps({"condition": self.condition}) + "\n", encoding="utf-8")


class ExplicitIpaModel(AdapterBase):
    def __init__(self, qwen: nn.Module, token_ids: list[int], fallback_token_id: int):
        super().__init__(qwen, "explicit_ipa")
        self.adapter = ExplicitIpaAdapter(
            token_ids,
            text_hidden_size(qwen.config),
            initializer_range=getattr(qwen.config, "initializer_range", 0.02),
        )
        self.adapter.to(dtype=next(qwen.parameters()).dtype)
        self.fallback_token_id = fallback_token_id

    def forward(self, input_ids, attention_mask, labels=None, **_):
        embeddings = self.adapter.apply(input_ids, self.qwen.get_input_embeddings(), self.fallback_token_id)
        return self.qwen(inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels, use_cache=False)

    def state_dict(self, *args, **kwargs):
        return {f"adapter.{name}": value for name, value in self.adapter.state_dict().items()}

    def load_adapter(self, path: Path) -> None:
        state = load_adapter_state(path)
        self.adapter.load_state_dict({name.removeprefix("adapter."): value for name, value in state.items() if name.startswith("adapter.")})


class FusionModel(AdapterBase):
    def __init__(self, qwen: nn.Module, xphonebert: nn.Module):
        super().__init__(qwen, "fusion")
        self.xphonebert = xphonebert
        freeze(xphonebert)
        qwen_hidden_size = text_hidden_size(qwen.config)
        self.resampler = PhonemeResampler(qwen_hidden_size, xphonebert.config.hidden_size)
        self.projector = PhoneticProjector(xphonebert.config.hidden_size, qwen_hidden_size)
        self.fusion = ResidualFusion()
        qwen_dtype = next(qwen.parameters()).dtype
        self.resampler.to(dtype=qwen_dtype)
        self.projector.to(dtype=qwen_dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.xphonebert.eval()
        return self

    def ipa_states(self, ipa_input_ids, ipa_attention_mask, ipa_window_token_index, ipa_window_example_index, ipa_sequence_lengths):
        with torch.no_grad():
            window_states = self.xphonebert(input_ids=ipa_input_ids, attention_mask=ipa_attention_mask).last_hidden_state
        batch_size = ipa_sequence_lengths.shape[0]
        maximum = int(ipa_sequence_lengths.max().item())
        hidden_size = window_states.shape[-1]
        valid = ipa_window_token_index >= 0
        example_indices = ipa_window_example_index[:, None].expand_as(ipa_window_token_index)
        destination = example_indices[valid] * maximum + ipa_window_token_index[valid]
        sums = torch.zeros(batch_size * maximum, hidden_size, dtype=torch.float32, device=window_states.device)
        counts = torch.zeros(batch_size * maximum, dtype=torch.float32, device=window_states.device)
        sums.index_add_(0, destination, window_states[valid].float())
        counts.index_add_(0, destination, torch.ones_like(destination, dtype=torch.float32))
        ipa_positions = torch.arange(maximum, device=window_states.device)[None, :] < ipa_sequence_lengths[:, None]
        coverage = counts.view(batch_size, maximum)
        if not torch.all(coverage[ipa_positions] > 0):
            raise ValueError("Chunked XPhoneBERT encoding dropped an IPA token")
        ipa = (sums / counts.clamp_min(1).unsqueeze(-1)).view(batch_size, maximum, hidden_size).to(window_states.dtype)
        return ipa, ipa_positions.to(dtype=ipa_attention_mask.dtype)

    def embeddings(self, input_ids, ipa_input_ids, ipa_attention_mask, ipa_window_token_index, ipa_window_example_index, ipa_sequence_lengths, variant_mask):
        with torch.no_grad():
            text = self.qwen.get_input_embeddings()(input_ids)
        ipa, ipa_attention_mask = self.ipa_states(
            ipa_input_ids, ipa_attention_mask, ipa_window_token_index, ipa_window_example_index, ipa_sequence_lengths
        )
        phonetic = self.projector(self.resampler(text, ipa, ipa_attention_mask))
        return self.fusion(text, phonetic, variant_mask)

    def forward(self, input_ids, attention_mask, ipa_input_ids, ipa_attention_mask, ipa_window_token_index, ipa_window_example_index, ipa_sequence_lengths, variant_mask, labels=None, **_):
        embeddings = self.embeddings(
            input_ids, ipa_input_ids, ipa_attention_mask, ipa_window_token_index, ipa_window_example_index,
            ipa_sequence_lengths, variant_mask
        )
        return self.qwen(inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels, use_cache=False)

    def state_dict(self, *args, **kwargs):
        result = {}
        for prefix, module in (("resampler", self.resampler), ("projector", self.projector), ("fusion", self.fusion)):
            result.update({f"{prefix}.{name}": value for name, value in module.state_dict().items()})
        return result

    def load_adapter(self, path: Path) -> None:
        state = load_adapter_state(path)
        for prefix, module in (("resampler", self.resampler), ("projector", self.projector), ("fusion", self.fusion)):
            module.load_state_dict({name.removeprefix(prefix + "."): value for name, value in state.items() if name.startswith(prefix + ".")})


def load_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    """Read either a compact final export or a Trainer epoch checkpoint."""
    if path.is_dir():
        compact = path / "adapter.pt"
        safe = path / "model.safetensors"
    else:
        compact, safe = path, None
    if compact.is_file():
        state = torch.load(compact, map_location="cpu", weights_only=True)
        if any(isinstance(value, dict) for value in state.values()):
            return {f"{prefix}.{name}": value for prefix, nested in state.items() for name, value in nested.items()}
        return state
    if safe and safe.is_file():
        from safetensors.torch import load_file

        return load_file(safe, device="cpu")
    raise FileNotFoundError(f"No adapter.pt or model.safetensors under {path}")
