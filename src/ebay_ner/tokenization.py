from typing import Any, Dict, List

from transformers import AutoTokenizer

from .config import BASE_MODEL_NAME, MAX_SEQ_LEN, WEAK_MODEL_DIR


def load_tokenizer(
    from_weak_model: bool = False,
):  # setup fallback if using base for experimentation
    if from_weak_model:
        return AutoTokenizer.from_pretrained(str(WEAK_MODEL_DIR))
    return AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)


def tok_fn_weak(batch: Dict[str, Any], tokenizer, max_length: int = MAX_SEQ_LEN):
    enc = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    all_labels = []
    for i in range(len(enc["input_ids"])):
        word_ids = enc.word_ids(batch_index=i)
        gold = batch["ner_tags"][i]
        seq = []
        prev = None
        for wid in word_ids:
            if wid is None:
                seq.append(-100)
            elif wid != prev:
                seq.append(gold[wid])
                prev = wid
            else:
                seq.append(-100)
        all_labels.append(seq)

    enc["labels"] = all_labels
    enc["category_id"] = batch["Category"]
    return enc


def tok_fn_supervised(batch: Dict[str, Any], tokenizer, max_length: int = MAX_SEQ_LEN):
    enc = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    all_labels = []
    word_ids_per_example: List[List[int | None]] = []
    for i in range(len(enc["input_ids"])):
        word_ids = enc.word_ids(batch_index=i)
        word_ids_per_example.append(word_ids)
        gold = batch["ner_tags"][i]
        seq = []
        prev = None
        for wid in word_ids:
            if wid is None:  # failing finetuning, bandaid solution
                seq.append(-100)
            elif wid != prev:
                seq.append(gold[wid])
                prev = wid
            else:
                seq.append(-100)
        all_labels.append(seq)

    enc["labels"] = all_labels
    enc["word_ids"] = word_ids_per_example
    enc["category_id"] = batch["Category"]
    return enc


def tok_quiz(batch: Dict[str, Any], tokenizer, max_length: int = MAX_SEQ_LEN):
    enc = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    enc["labels"] = [[-100] * len(ids) for ids in enc["input_ids"]]
    enc["word_ids"] = [enc.word_ids(i) for i in range(len(enc["input_ids"]))]
    enc["category_id"] = batch["Category"]
    enc["record_id"] = batch["Record Number"]
    enc["tokens"] = batch["tokens"]
    return enc
