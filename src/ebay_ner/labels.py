from typing import Dict, List, Tuple

import pandas as pd


def build_bio_labels(
    tagged_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Build BIO label list + mappings from the tagged training dataframe.
    """
    base_tags = {t for t in tagged_df["Tag"].unique() if t and t != "O"}
    label_list = ["O"] + sorted(f"{p}-{t}" for t in base_tags for p in ("B", "I"))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label


def build_allow_mask(tagged_df: pd.DataFrame, label_list):
    """
    Per-category allow mask: for each category, which labels are allowed.
    """
    import torch

    allow = (
        tagged_df.groupby("Category")["Tag"]
        .apply(lambda s: set(s.unique()) - {""})
        .to_dict()
    )

    allow_mask = {}
    for cat, tags in allow.items():
        ok = {"O"}
        for t in tags:
            ok.add(f"B-{t}")
            ok.add(f"I-{t}")
        allow_mask[int(cat)] = torch.tensor([l in ok for l in label_list])

    return allow_mask, allow
