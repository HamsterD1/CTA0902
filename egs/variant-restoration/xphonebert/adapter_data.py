"""Data encoding for the Explicit IPA and XPhoneBERT fusion adapters."""

from __future__ import annotations

from data import encode_supervised, messages, prompt_ids
from xphonebert_chunks import chunk_ipa


def variant_mask(tokenizer, descriptor: dict, row: dict, ipa_section: str, prompt: list[int]) -> list[int]:
    rendered = tokenizer.apply_chat_template(messages(descriptor, row, ipa_section), tokenize=False, add_generation_prompt=True)
    user = messages(descriptor, row, ipa_section)[-1]["content"]
    user_start = rendered.find(user)
    variant_start = user_start + user.find(row["variant_text"])
    if user_start < 0 or variant_start < user_start:
        raise ValueError(f"Could not locate user content in rendered chat template for {row['id']}")
    variant_end = variant_start + len(row["variant_text"])
    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if tokenized["input_ids"] != prompt:
        raise ValueError(f"Chat-template tokenization mismatch for {row['id']}")
    return [int(start < variant_end and end > variant_start) for start, end in tokenized["offset_mapping"]]


class AdapterDataset:
    def __init__(self, rows, tokenizer, descriptor, condition: str, ipa_token_map: dict[str, str], max_length: int):
        self.rows = rows
        self.examples = []
        for row in rows:
            ipa_section = ""
            if condition == "explicit_ipa":
                ipa_section = "\n分段 IPA：" + " ".join(ipa_token_map[unit] for unit in row["ipa"].split())
            item = encode_supervised(tokenizer, descriptor, row, ipa_section)
            if len(item["input_ids"]) > max_length:
                raise ValueError(f"{row['id']} exceeds fixed max length; refusing to truncate")
            item["variant_mask"] = variant_mask(tokenizer, descriptor, row, ipa_section, item["input_ids"][:item["prompt_length"]])
            item["variant_mask"] += [0] * (len(item["input_ids"]) - item["prompt_length"])
            item["ipa"] = row["ipa"]
            self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class AdapterCollator:
    def __init__(self, qwen_tokenizer, xphonebert_tokenizer, condition: str):
        self.pad_id = qwen_tokenizer.pad_token_id if qwen_tokenizer.pad_token_id is not None else qwen_tokenizer.eos_token_id
        self.xphonebert_tokenizer = xphonebert_tokenizer
        self.condition = condition

    def _chunked_ipa(self, features):
        import torch

        chunks = []
        sequence_lengths = []
        for example_index, row in enumerate(features):
            plan = chunk_ipa(self.xphonebert_tokenizer, row["ipa"])
            sequence_lengths.append(plan.sequence_token_count)
            for window in plan.windows:
                chunks.append((example_index, window))
        if not chunks:
            raise ValueError("Fusion batch contains no IPA windows")
        maximum = max(len(window.input_ids) for _, window in chunks)
        pad_id = self.xphonebert_tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("XPhoneBERT tokenizer requires a pad token")
        ipa_input_ids, ipa_attention_mask, ipa_window_token_index, ipa_window_example_index = [], [], [], []
        for example_index, window in chunks:
            padding = maximum - len(window.input_ids)
            original_indices = [-1] * maximum
            for position, original_index in zip(window.content_positions, window.original_token_indices, strict=True):
                original_indices[position] = original_index
            ipa_input_ids.append(window.input_ids + [pad_id] * padding)
            ipa_attention_mask.append([1] * len(window.input_ids) + [0] * padding)
            ipa_window_token_index.append(original_indices)
            ipa_window_example_index.append(example_index)
        return {
            "ipa_input_ids": torch.tensor(ipa_input_ids, dtype=torch.long),
            "ipa_attention_mask": torch.tensor(ipa_attention_mask, dtype=torch.long),
            "ipa_window_token_index": torch.tensor(ipa_window_token_index, dtype=torch.long),
            "ipa_window_example_index": torch.tensor(ipa_window_example_index, dtype=torch.long),
            "ipa_sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        }

    def __call__(self, features):
        import torch

        maximum = max(len(row["input_ids"]) for row in features)
        result = {"input_ids": [], "labels": [], "attention_mask": [], "variant_mask": []}
        for row in features:
            padding = maximum - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_id] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
            result["attention_mask"].append([1] * len(row["input_ids"]) + [0] * padding)
            result["variant_mask"].append(row["variant_mask"] + [0] * padding)
        result = {name: torch.tensor(value, dtype=torch.long) for name, value in result.items()}
        if self.condition == "fusion":
            result.update(self._chunked_ipa(features))
        return result
