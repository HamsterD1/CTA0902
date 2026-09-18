"""Frozen-Qwen adapter wrappers with compact checkpoint state dicts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from modeling import ExplicitIpaAdapter, PhonemeResampler, PhoneticProjector, ResidualFusion


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

    def save_pretrained(self, directory: str | Path, **_: object) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "adapter.pt")
        (directory / "adapter_config.json").write_text(json.dumps({"condition": self.condition}) + "\n", encoding="utf-8")


class ExplicitIpaModel(AdapterBase):
    def __init__(self, qwen: nn.Module, token_ids: list[int], fallback_token_id: int):
        super().__init__(qwen, "explicit_ipa")
        self.adapter = ExplicitIpaAdapter(token_ids, qwen.config.hidden_size)
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
        self.resampler = PhonemeResampler(qwen.config.hidden_size, xphonebert.config.hidden_size)
        self.projector = PhoneticProjector(xphonebert.config.hidden_size, qwen.config.hidden_size)
        self.fusion = ResidualFusion()

    def embeddings(self, input_ids, ipa_input_ids, ipa_attention_mask, variant_mask):
        with torch.no_grad():
            text = self.qwen.get_input_embeddings()(input_ids)
            ipa = self.xphonebert(input_ids=ipa_input_ids, attention_mask=ipa_attention_mask).last_hidden_state
        phonetic = self.projector(self.resampler(text, ipa, ipa_attention_mask))
        return self.fusion(text, phonetic, variant_mask)

    def forward(self, input_ids, attention_mask, ipa_input_ids, ipa_attention_mask, variant_mask, labels=None, **_):
        embeddings = self.embeddings(input_ids, ipa_input_ids, ipa_attention_mask, variant_mask)
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
