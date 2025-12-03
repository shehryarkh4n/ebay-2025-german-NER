from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, PreTrainedModel

# from .config import BASE_MODEL_NAME, WEAK_MODEL_DIR


class WeakNERModel(PreTrainedModel):
    """Simple DeBERTa encoder + linear head for weak pretraining."""

    config_class = AutoConfig

    def __init__(self, config, num_labels: int, base_model_name: str | None = None):
        super().__init__(config)
        self.num_labels = num_labels

        if base_model_name:
            self.encoder = AutoModel.from_pretrained(base_model_name)
        else:
            self.encoder = AutoModel.from_config(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(sequence_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}


class CatAwareCRF(PreTrainedModel):
    """
    Encoder + category embedding + projection + CRF.
    Uses per-category allow_mask to hard-mask impossible labels.
    """

    config_class = AutoConfig

    def __init__(
        self,
        config,
        num_labels: int,
        allow_mask: Dict[int, torch.Tensor] | None = None,
        base_model_name: str | None = None,
        use_dapt: bool = False,
        label_smoothing: float = 0.02,
        **kwargs,
    ):
        super().__init__(config)
        self.num_labels = num_labels
        self.allow_mask = {k: v.bool() for k, v in (allow_mask or {}).items()}
        self.label_smoothing = label_smoothing

        if use_dapt and base_model_name:
            print(f"Loading encoder from {base_model_name}")
            self.encoder = AutoModel.from_pretrained(base_model_name)
        elif base_model_name:
            self.encoder = AutoModel.from_pretrained(base_model_name)
        else:
            self.encoder = AutoModel.from_config(config)

        self.cat_embed = nn.Embedding(3, 64)
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(config.hidden_size + 64, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self._init_task_layers()

    def _init_task_layers(self):
        nn.init.normal_(self.cat_embed.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        category_id=None,
        **ignored,  # unwrap as needed
    ):
        h = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        cat = self.cat_embed(category_id).unsqueeze(1).expand(-1, h.size(1), -1)
        combined = torch.cat([h, cat], dim=-1)

        logits = self.proj(self.dropout(combined))

        # Category-aware masking
        for c, mask in self.allow_mask.items():
            bad = ~mask.to(logits.device)
            idx = (category_id == c).nonzero(as_tuple=True)[0]
            if len(idx):
                logits[idx][:, :, bad] = -1e10

        if labels is not None:
            mask = attention_mask.bool()
            safe_labels = labels.clone()
            safe_labels[labels == -100] = 0

            if self.label_smoothing > 0:
                log_lik = self.crf(
                    logits, safe_labels, mask=mask, reduction="token_mean"
                )
                num_labels = logits.size(-1)
                uniform_dist = torch.full_like(logits, 1.0 / num_labels)
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    uniform_dist,
                    reduction="batchmean",
                )
                loss = -log_lik + self.label_smoothing * kl_loss
            else:
                log_lik = self.crf(
                    logits, safe_labels, mask=mask, reduction="token_mean"
                )
                loss = -log_lik

            return {"loss": loss, "logits": logits}
        else:
            paths = self.crf.decode(logits, mask=attention_mask.bool())
            max_len = logits.size(1)
            out = torch.full(
                (len(paths), max_len),
                -100,
                dtype=torch.long,
                device=logits.device,
            )
            for i, seq in enumerate(paths):
                out[i, : len(seq)] = torch.tensor(seq, device=logits.device)
            return {"logits": out}
