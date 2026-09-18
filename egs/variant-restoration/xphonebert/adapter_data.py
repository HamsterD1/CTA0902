"""Data encoding for the Explicit IPA and XPhoneBERT fusion adapters."""

from __future__ import annotations

from data import encode_supervised, messages, prompt_ids


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
            ipa = self.xphonebert_tokenizer([row["ipa"] for row in features], padding=True, return_tensors="pt")
            result["ipa_input_ids"] = ipa["input_ids"]
            result["ipa_attention_mask"] = ipa["attention_mask"]
        return result
