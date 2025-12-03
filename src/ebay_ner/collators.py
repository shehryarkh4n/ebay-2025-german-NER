from typing import Any, Dict, List

import torch


def new_collator(
    features: List[Dict[str, Any]], pad_token_id: int
):  # updated collator.
    max_len = max(len(f["input_ids"]) for f in features)

    def pad_to_max(seq, pad_value=0):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = torch.tensor(
        [pad_to_max(f["input_ids"], pad_value=pad_token_id) for f in features],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [pad_to_max(f["attention_mask"], pad_value=0) for f in features],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [pad_to_max(f["labels"], pad_value=-100) for f in features],
        dtype=torch.long,
    )
    category_id = torch.tensor([f["category_id"] for f in features], dtype=torch.long)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "category_id": category_id,
    }

    if "word_ids" in features[0]:
        batch["word_ids"] = [f["word_ids"] for f in features]

    return batch
