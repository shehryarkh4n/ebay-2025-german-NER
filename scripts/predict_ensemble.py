"""
running predictions w/ ensemble inference on Listing_Titles.tsv quiz split and write TSV submission.
"""

import csv

import pandas as pd
from datasets import Dataset

from ebay_ner.config import DATA_DIR, RESULTS_DIR
from ebay_ner.ensemble import ensemble_predict, load_ensemble_models
from ebay_ner.labels import build_allow_mask, build_bio_labels
from ebay_ner.postprocess import post_process_predictions
from ebay_ner.tokenization import load_tokenizer, tok_quiz
from ebay_ner.weak_labeling import build_gazetteers


def main():
    # Load train to restore labels & allow sets
    tagged = pd.read_csv(
        DATA_DIR / "Tagged_Titles_Train.tsv",
        sep="\t",
        keep_default_na=False,
        na_values=None,
    )
    label_list, label2id, id2label = build_bio_labels(tagged)
    allow_mask, allow = build_allow_mask(tagged, label_list)
    gazetteers = build_gazetteers(tagged)

    # quiz data
    quiz = (
        pd.read_csv(
            DATA_DIR / "Listing_Titles.tsv",
            sep="\t",
            keep_default_na=False,
            na_values=None,
        )
        .query("5001 <= `Record Number` <= 30000")  # Ebay directed range for quiz data
        .copy()
    )

    quiz["tokens"] = quiz["Title"].str.split()
    quiz_ds = Dataset.from_pandas(
        quiz[["Record Number", "Category", "tokens"]], preserve_index=False
    )

    tokenizer = load_tokenizer(from_weak_model=True)
    tokenised_quiz = quiz_ds.map(
        lambda batch: tok_quiz(batch, tokenizer),
        batched=True,
        remove_columns=[],
    )

    # seeds from training script
    seeds = [42, 123, 456, 789, 2024]
    ensemble_paths = [
        str((RESULTS_DIR.parent / "models" / f"deberta-ner-ensemble-seed{s}-final"))
        for s in seeds
    ]

    models = load_ensemble_models(
        model_paths=ensemble_paths, allow_mask=allow_mask, num_labels=len(label_list)
    )
    pred_ids = ensemble_predict(models, tokenised_quiz, tokenizer)
    print("Ensemble predictions shape:", pred_ids.shape)

    records = []
    for i, ex in enumerate(tokenised_quiz):
        rec = int(ex["record_id"])
        cat = int(ex["category_id"])
        words = ex["tokens"]
        wids = ex["word_ids"]
        labs = [id2label[idx] if idx != -100 else "O" for idx in pred_ids[i]]

        word_labels = []
        prev_wid = None
        for wid, lab in zip(wids, labs):
            if wid is not None and wid != prev_wid:
                word_labels.append((wid, lab))
                prev_wid = wid

        current_tokens = []
        current_tag = None

        for wid, label in word_labels:
            word = words[wid]
            if label == "O":
                if current_tokens and current_tag:
                    records.append((rec, cat, current_tag, " ".join(current_tokens)))
                    current_tokens = []
                    current_tag = None
                records.append((rec, cat, "O", word))
                continue

            prefix, tag = label.split("-", 1)
            if tag not in allow[cat]:
                continue

            if prefix == "B":
                if current_tokens and current_tag:
                    records.append((rec, cat, current_tag, " ".join(current_tokens)))
                current_tokens = [word]
                current_tag = tag
            elif prefix == "I":
                if tag == current_tag:
                    current_tokens.append(word)
                else:
                    if current_tokens and current_tag:
                        records.append(
                            (rec, cat, current_tag, " ".join(current_tokens))
                        )
                    current_tokens = [word]
                    current_tag = tag

        if current_tokens and current_tag:
            records.append((rec, cat, current_tag, " ".join(current_tokens)))

    # Post-process
    records = post_process_predictions(records, gazetteers, allow)
    submission = pd.DataFrame(
        records, columns=["Record Number", "Category", "Tag", "Token"]
    )
    submission = submission[submission["Tag"] != "O"]

    # for my own sanity, check if there's mismatching category for any row
    # This auto failed my earlier results in ebay's official scorer
    quiz_categories = quiz.set_index("Record Number")["Category"].to_dict()
    for idx, row in submission.iterrows():
        rec_num = row["Record Number"]
        if row["Category"] != quiz_categories.get(rec_num):
            print(f"WARN: Category mismatch at record {rec_num}!")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        RESULTS_DIR / "EXAMPLE_NAME_SUBMIT.tsv"
    )  #! Change file submission name here. Too lazy to code out_name as an arg.
    submission.to_csv(
        out_path,
        sep="\t",
        header=False,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    print("Saved submission to", out_path)


if __name__ == "__main__":
    main()
