# eBay German NER 2025 Challenge

Link to (finished) competition: [EvalAI](https://eval.ai/web/challenges/challenge-page/2508/overview)

**A two-stage NER system for automotive product titles, featuring gazetteer-enhanced weak labeling, DeBERTa-V3 pretraining, and a CRF ensemble.**

This repository hosts my complete pipeline for the **eBay German Named Entity Recognition challenge**, where the official eBay team baseline scored **~80%**, and this approach achieved **~87.37%** (see `results/87-37pc.tsv`).

This repo exists as a sanity check for myself, porting over all my roughly done jupyter work into a modularized pipeline that makes more sense.

If you're here to run everything from scratch… **I wish you good luck and an A30**.


## But TL;DR: What’s inside?

* **src/ebay_ner/**
  The actual implementation:

  * **Improved weak labeling system** (context rules, heuristics, patterns)
  * **Weak NER DeBERTa pretraining** (replaces MLM pretraining, which was my first move)
  * **Supervised CRF head with category masking**
  * **Ensemble training + majority voting inference**
  * **Post-processing sanity rules** (because models hallucinate, and I get scared)

* **scripts/**

  * `train_weak.py` — Build weak labels and pretrain the encoder
  * `train_ensemble.py` — Fine-tune 5 independently seeded CRF models
  * `predict_ensemble.py` — Run ensemble inference, decode, post-process, output the TSV

* **results/**

  * Contains **`87-37pc.tsv`** — my best submission (87.37% F1_beta)

* **old_notebooks/**

  * The archaeological layer of this project.
    Where all sins were committed and later forgiven.
    This acts as a basis from which I rewrote code into the current pipeline.


## High-Level Architecture

```
Unlabeled Titles ──┐
                   │   Gazetteers + Contextual Rules + Regex Patterns
                   └── Weak Labeler  ───►  150k–200k Weak NER Dataset
                                            │
                                            ▼
                                 DeBERTa-V3-Large (Weak Pretrained)
                                            │
                                            ▼
             Tagged Train Data ──►  Category-Aware CRF Fine-Tuning (x5 seeds)
                                            │
                                            ▼
                                   Ensemble Majority Voting
                                            │
                                            ▼
                                 Post-Process + Output TSV
```

---

## Results

**Best score achieved: ~87.37%**, stored as:

```
results/87-37pc.tsv
```

That file reflects the full pipeline: weak pretraining → CRF fine-tuning → ensemble voting → post-processing.

---

## Why it works

* **Weak labeling is great** when done carefully.
  Context rules like *“für BMW” = compatible brand* vs *“ORIGINAL BMW” = manufacturer* help a lot.

* **Weak NER pretraining** with DeBERTa-V3-Large gives a huge boost before supervised finetuning.

* **Category-aware CRF** disallows illegal tags (Category 1 and Category 2 have different valid entities). This was actually a huge problem early on. Ebay's internal scorer broke on encountering wrong tags in submitted files.

* **Ensemble** of 5 independently seeded models improves stability and F1. More did not equal better. My guess is to, tip over 90% (which teams did manage), you'd have to manually label more data, try multiple start-stops, and different architectures as different ensembles. Rest assured, one can imagine even 87% can be achieved with the most "auto-pilot" strategies 

* **Post-processing** catches common sense mistakes (BMW is not a manufacturer… mostly).

---

## How to Run (in theory)

This repo is mainly to **preserve the cleaned pipeline**, not necessarily to encourage full reproduction.
But here you go:

### 1. Weak Label Pretraining

```bash
python scripts/train_weak.py
```

This:

* builds gazetteers,
* generates ~200k weak NER samples in parallel,
* trains a DeBERTa model for 2–3 epochs,
* saves the encoder to `models/deberta-improved-weak-ner-mk-2`.

### 2. CRF Fine-tuning + Ensemble

```bash
python scripts/train_ensemble.py
```

Trains **five** models: seeds `[42, 123, 456, 789, 2024]`.

### 3. Ensemble Inference

```bash
python scripts/predict_ensemble.py
```

Produces a TSV similar to the final one in `results/`.

---

## Folder Structure

```
...
│
├─ src/ebay_ner/
│   ├─ weak_labeling.py        # Gazetteers + contextual weak labeling
│   ├─ models.py               # WeakNERModel + CatAwareCRF
│   ├─ ensemble.py             # Train multiple models + voting
│   ├─ tokenization.py         # NER tokenization helpers
│   ├─ postprocess.py          # Fix mistakes after decoding
│   ├─ labels.py               # BIO labels + allow mask logic
│   ├─ config.py               # Paths and constants
│   └─ utils.py
│
├─ scripts/
│   ├─ train_weak.py
│   ├─ train_ensemble.py
│   └─ predict_ensemble.py
│
├─ data/                       # Not included here (TSVs from challenge)
│
├─ results/
│   └─ 87-37pc.tsv             # BEST RESULT OUTPUT
│   └─ 87-37pc-..._context.png # Ebay scorer output img
├─ old_notebooks/              # old rough work for transparency
└─ README.md
```


* The pipeline is modular, testable, and clearly separable into stages.
* The weak labeling component is explainable and domain-driven.
* The CRF implementation includes:

  * category-aware masking,
  * label smoothing,
  * ensemble voting for stability.
* The code can be run on a single A30, but can definitely be optimized further for lower VRAM compute.

---
Prior to working on this challenge, I had z-e-r-0 NER experience, and about as much experience with German as a non-German. Fun journey though.

It now lives here in a much more respectable, production-adjacent form.
