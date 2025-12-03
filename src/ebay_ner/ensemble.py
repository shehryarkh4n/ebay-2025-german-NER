from collections import Counter
from typing import List

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
)

from .collators import new_collator
from .config import ENSEMBLE_MODEL_PREFIX, MODELS_DIR, WEAK_MODEL_DIR
from .models import CatAwareCRF
from .utils import get_device, set_seed


def train_single_model(
    seed: int,
    train_dataset: Dataset,
    label_list,
    id2label,
    label2id,
    allow_mask,
    tokenizer,
):
    print("\n" + "=" * 60)
    print(f"Training Model seed={seed}")
    print("=" * 60 + "\n")

    set_seed(seed)
    device = get_device()

    cfg = AutoConfig.from_pretrained(
        str(WEAK_MODEL_DIR),
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    model = CatAwareCRF(
        cfg,
        num_labels=len(label_list),
        allow_mask=allow_mask,
        base_model_name=str(WEAK_MODEL_DIR),
        use_dapt=True,
        label_smoothing=0.005,
    ).to(device)

    args = TrainingArguments(
        output_dir=str(MODELS_DIR / f"{ENSEMBLE_MODEL_PREFIX}{seed}"),
        per_device_train_batch_size=48,
        gradient_accumulation_steps=3,
        learning_rate=4e-6,
        weight_decay=0.01,
        warmup_ratio=0.15,
        max_grad_norm=1.0,
        num_train_epochs=45,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine_with_restarts",
        bf16=True,
        fp16=False,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=25,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=lambda feats: new_collator(feats, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    trainer.train()
    final_dir = MODELS_DIR / f"{ENSEMBLE_MODEL_PREFIX}{seed}-final"
    trainer.save_model(str(final_dir))
    return final_dir


def load_ensemble_models(model_paths: List[str], allow_mask, num_labels: int):
    device = get_device()
    models = []
    for path in model_paths:
        print(f"Loading model from {path}...")
        cfg = AutoConfig.from_pretrained(path)
        model = CatAwareCRF(
            cfg,
            num_labels=num_labels,
            allow_mask=allow_mask,
            base_model_name=None,
            use_dapt=True,
        )
        state_dict = load_file(f"{path}/model.safetensors")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models, tokenised_quiz: Dataset, tokenizer):
    device = get_device()

    dummy_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "temp"),
        per_device_eval_batch_size=48,
        dataloader_num_workers=16,
        bf16=True,
        fp16=False,
        report_to="none",
    )

    all_predictions = []

    for i, model in enumerate(models):
        print(f"Predicting with model {i + 1}/{len(models)}...")

        trainer = Trainer(
            model=model,
            args=dummy_args,
            data_collator=lambda feats: new_collator(feats, tokenizer.pad_token_id),
            processing_class=tokenizer,
        )

        pred_output = trainer.predict(tokenised_quiz)
        pred_array = pred_output.predictions

        if pred_array.ndim == 3:
            pred_ids = pred_array.argmax(-1)
        elif pred_array.ndim == 2:
            pred_ids = pred_array
        else:
            raise ValueError(f"Unexpected prediction shape: {pred_array.shape}")

        all_predictions.append(pred_ids)
        torch.cuda.empty_cache()

    print("Performing majority voting...")
    all_predictions = np.array(all_predictions)  # (num_models, batch, seq_len)

    voted_predictions = []
    for i in range(all_predictions.shape[1]):  # per example
        example_preds = all_predictions[:, i, :]
        voted_seq = []
        for j in range(example_preds.shape[1]):
            token_votes = example_preds[:, j]
            valid = token_votes[token_votes != -100]
            if len(valid) == 0:
                voted_seq.append(-100)
            else:
                counts = Counter(valid)
                majority = counts.most_common(1)[0][0]
                voted_seq.append(int(majority))
        voted_predictions.append(voted_seq)

    return np.array(voted_predictions, dtype=np.int64)
