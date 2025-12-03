"""
Phase 1 - Weak NER pretraining, to warm up the model to nomenclature.

1. Load tagged + unlabeled data
2. Build gazetteers and BIO labels
3. Generate weakly-labeled HF Dataset
4. Train WeakNERModel on weak labels (w/ the new architecture)
5. Save encoder + tokenizer
"""

from datasets import DatasetDict
from transformers import (
    AutoConfig,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ebay_ner.config import MAX_SEQ_LEN, MODELS_DIR
from ebay_ner.models import WeakNERModel
from ebay_ner.tokenization import load_tokenizer, tok_fn_weak
from ebay_ner.utils import get_device
from ebay_ner.weak_labeling import (
    create_weak_ner_dataset_parallel,
    load_tagged_and_unlabeled,
)


def main():
    device = get_device()

    _, unlabeled, (label_list, label2id, id2label), gazetteers = (
        load_tagged_and_unlabeled()
    )

    weak_ds = create_weak_ner_dataset_parallel(
        unlabeled_df=unlabeled,
        gazetteers=gazetteers,
        label2id=label2id,
        sample_size=200_000,
    )

    weak_splits = weak_ds.train_test_split(test_size=0.05, seed=42)
    weak_splits = DatasetDict(
        {"train": weak_splits["train"], "validation": weak_splits["test"]}
    )

    tokenizer = load_tokenizer(from_weak_model=False)

    tok_weak_train = weak_splits["train"].map(
        lambda batch: tok_fn_weak(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags", "Category"],
        num_proc=16,
    )

    tok_weak_val = weak_splits["validation"].map(
        lambda batch: tok_fn_weak(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags", "Category"],
        num_proc=16,
    )

    cfg = AutoConfig.from_pretrained(
        tokenizer.name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    model = WeakNERModel(
        cfg, num_labels=len(label_list), base_model_name=tokenizer.name_or_path
    ).to(device)

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, max_length=MAX_SEQ_LEN, pad_to_multiple_of=8
    )

    out_dir = MODELS_DIR / "deberta-improved-weak-ner-mk-2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # no reporting, basic low lr pre-training
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=48,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_epochs=3,
        optim="adamw_torch_fused",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        logging_steps=100,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_weak_train,
        eval_dataset=tok_weak_val,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("Starting weak NER pre-training...")
    trainer.train()

    model.encoder.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Weak NER encoder saved at", out_dir)


if __name__ == "__main__":
    main()
