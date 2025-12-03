"""
phase 2: supervised fine-tuning + ensemble.

1. Load tagged train TSV
2. Convert to HF Dataset (rows_to_examples)
3. Tokenize using weak-pretrained tokenizer
4. Train multiple CatAwareCRF models with different seeds
"""

import pandas as pd
import torch
from datasets import Dataset

from ebay_ner.config import DATA_DIR
from ebay_ner.ensemble import train_single_model
from ebay_ner.labels import build_allow_mask, build_bio_labels
from ebay_ner.tokenization import load_tokenizer, tok_fn_supervised


def rows_to_examples(df: pd.DataFrame, label2id):
    """
    multi-token entities and separate consecutive same-tag entities.
    """
    records = []

    for record_num, group in df.groupby("Record Number"):
        group = group.sort_index()
        tokens = []
        bio_tags = []
        prev_tag = None

        for _, row in group.iterrows():
            token = row["Token"]
            tag = row["Tag"]

            if pd.isna(tag) or tag == "":
                if prev_tag and prev_tag != "O":
                    tokens.append(token)
                    bio_tags.append(f"I-{prev_tag}")
                else:
                    tokens.append(token)
                    bio_tags.append("O")
                    prev_tag = "O"
                continue

            tokens.append(token)
            if tag == "O":
                bio_tags.append("O")
                prev_tag = "O"
            else:
                bio_tags.append(f"B-{tag}")
                prev_tag = tag

        records.append(
            {
                "tokens": tokens,
                "ner_tags": [label2id[b] for b in bio_tags],
                "Category": int(group["Category"].iloc[0]),
            }
        )

    return records


def main():
    tagged = pd.read_csv(
        DATA_DIR / "Tagged_Titles_Train.tsv",
        sep="\t",
        keep_default_na=False,
        na_values=None,
    )

    label_list, label2id, id2label = build_bio_labels(tagged)
    allow_mask, _ = build_allow_mask(tagged, label_list)

    examples = rows_to_examples(tagged, label2id)
    hf_ds = Dataset.from_list(examples)

    tokenizer = load_tokenizer(from_weak_model=True)
    tokenised = hf_ds.map(
        lambda batch: tok_fn_supervised(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags", "Category"],
    )

    ensemble_seeds = [42, 123, 456, 789, 2024]  # try upto 15
    ensemble_paths = []

    for seed in ensemble_seeds:
        model_path = train_single_model(
            seed=seed,
            train_dataset=tokenised,
            label_list=label_list,
            id2label=id2label,
            label2id=label2id,
            allow_mask=allow_mask,
            tokenizer=tokenizer,
        )
        ensemble_paths.append(model_path)

        torch.cuda.empty_cache()

    print(f"\nTrained {len(ensemble_paths)} models")
    for p in ensemble_paths:
        print("  -", p)


if __name__ == "__main__":
    main()
