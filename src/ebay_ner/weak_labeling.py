import re
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

from .config import DATA_DIR
from .labels import build_bio_labels


class ImprovedWeakNERLabeler:
    """
    Supposedly enhanced weak labeling with:
      1. Context-aware rules (für X vs ORIGINAL X)
      2. Pattern matching (dimensions, teeth, positions)
      3. Brand/manufacturer heuristics
      Hope this works better
    """

    def __init__(self, gazetteers: Dict[str, set], category: int):
        self.gazetteers = gazetteers
        self.category = category

        # disambiguation
        self.context_rules = {
            "Kompatible_Fahrzeug_Marke": {
                "before": ["für", "passend", "geeignet", "kompatibel"],
                "avoid_before": ["original", "oem", "genuine"],
            },
            "Hersteller": {
                "before": ["original", "oem", "genuine", "von"],
                "after": ["marke", "qualität", "hersteller"],
            },
        }

        # regex patterns for automotive-ish entities
        self.patterns = {
            "Anzahl_Der_Einheiten": [
                (r"\b(\d+)\s*(stück|stk|x|pcs|piece|teilig)\b", 1),
                (r"\b(\d+)er[\s-]*(set|satz)\b", 1),
                (r"\b(\d+)[\s-]*teilig\b", 1),
            ],
            "Einbauposition": [
                (r"\b(vorne?|hinten?|links?|rechts?|va|ha|vl|vr|hl|hr)\b", 0),
                (r"\b(vorder|hinter|vorn|hint)\w*\b", 0),
                (r"\b(front|rear|left|right)\b", 0),
            ],
            "Durchmesser": [
                (r"\b(ø|durchmesser|dm\.?|diameter)\s*(\d+)\s*(mm|cm)?\b", 2),
                (r"\b(\d+)\s*mm\b", 1),
            ],
            "Breite": [
                (r"\bbreite\s*(\d+)\s*(mm|cm)?\b", 1),
                (r"\b(\d+)\s*mm\s*breit\b", 1),
            ],
            "Zähnezahl": [
                (r"\b(\d+)\s*zähne?\b", 1),
                (r"\b(\d+)[\s-]*teeth\b", 1),
            ],
            "Kompatibles_Fahrzeug_Modell": [
                (r"\b([A-Z]\d+)\b", 0),
                (r"\b([A-Z]{1,3}[\s-]\d{1,3})\b", 0),
            ],
        }

        # Known brands / manufacturers.
        # TODO: Manual check for more
        self.known_brands = {
            "bmw",
            "audi",
            "vw",
            "volkswagen",
            "mercedes",
            "benz",
            "opel",
            "ford",
            "renault",
            "peugeot",
            "citroen",
            "fiat",
            "seat",
            "skoda",
            "porsche",
            "volvo",
            "saab",
            "toyota",
            "nissan",
            "honda",
            "mazda",
            "hyundai",
            "kia",
            "chevrolet",
            "chrysler",
            "jeep",
            "land",
            "rover",
            "mini",
            "alfa",
            "romeo",
            "lancia",
            "subaru",
            "suzuki",
            "mitsubishi",
            "dacia",
            "jaguar",
        }

        self.known_manufacturers = {
            "bosch",
            "ate",
            "brembo",
            "zimmermann",
            "febi",
            "lemförder",
            "sachs",
            "bilstein",
            "corteco",
            "mahle",
            "mann",
            "hella",
            "valeo",
            "continental",
            "trw",
            "skf",
            "fag",
            "snr",
            "ina",
            "dayco",
            "gates",
            "contitech",
            "meyle",
            "optimal",
            "ruville",
            "swag",
            "topran",
            "trucktec",
            "vemo",
            "pierburg",
            "elring",
        }

    def label_text(self, text: str) -> List[Tuple[str, str]]:
        """
        returns list of (token, BIO-tag) pairs with context-aware labeling.
        """
        tokens = text.split()
        labels = ["O"] * len(tokens)
        text_lower = text.lower()

        # 1: gazetteer matching
        for tag, entities in self.gazetteers.items():
            for entity in entities:
                entity_lower = entity.lower()

                if " " in entity_lower:
                    # Multi-token entity
                    entity_tokens = entity_lower.split()
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        window = " ".join(tokens[i : i + len(entity_tokens)]).lower()
                        if window == entity_lower:
                            resolved_tag = self._resolve_tag_with_context(
                                tag, i, tokens, entity_lower
                            )
                            if labels[i] == "O":
                                labels[i] = f"B-{resolved_tag}"
                                for j in range(i + 1, i + len(entity_tokens)):
                                    labels[j] = f"I-{resolved_tag}"
                else:
                    # Single-token entity
                    for i, tok in enumerate(tokens):
                        if tok.lower() == entity_lower and labels[i] == "O":
                            resolved_tag = self._resolve_tag_with_context(
                                tag, i, tokens, entity_lower
                            )
                            labels[i] = f"B-{resolved_tag}"

        # 2: brand / manufacturer heuristics
        for i, tok in enumerate(tokens):
            tok_lower = tok.lower().strip(".,;:")
            if labels[i] != "O":
                continue

            if tok_lower in self.known_brands:
                if i > 0:
                    prev = tokens[i - 1].lower()
                    if prev in ["für", "passend", "geeignet", "kompatibel"]:
                        labels[i] = "B-Kompatible_Fahrzeug_Marke"
                    elif prev in ["original", "oem", "genuine"]:
                        labels[i] = "B-Hersteller"
                    else:
                        labels[i] = "B-Kompatible_Fahrzeug_Marke"
                else:
                    labels[i] = "B-Kompatible_Fahrzeug_Marke"

            elif tok_lower in self.known_manufacturers:
                labels[i] = "B-Hersteller"

        # 3: pattern matching
        for tag, patterns in self.patterns.items():
            for pattern, group_idx in patterns:
                for match in re.finditer(pattern, text_lower, flags=re.IGNORECASE):
                    start_char = match.start(group_idx)
                    char_pos = 0
                    for i, tok in enumerate(tokens):
                        tok_len = len(tok)
                        if char_pos <= start_char < char_pos + tok_len:
                            if labels[i] == "O":
                                labels[i] = f"B-{tag}"
                            break
                        char_pos += tok_len + 1

        # 4: fix compound entities and apply contextual tagging
        labels = self._fix_compound_entities(tokens, labels)
        labels = self._contextual_tagging(tokens, labels)
        return list(zip(tokens, labels))

    # helpers

    def _contextual_tagging(self, tokens, labels):
        """Add context-aware corrections after initial tagging."""
        for i in range(len(tokens)):
            tok_lower = tokens[i].lower()

            # "für X"
            if i > 0 and tokens[i - 1].lower() == "für":
                if tok_lower in self.known_brands and labels[i] == "O":
                    labels[i] = "B-Kompatible_Fahrzeug_Marke"

            # "ORIGINAL X"
            if i > 0 and tokens[i - 1].lower() == "original":
                if labels[i] == "O" or labels[i].endswith("Kompatible_Fahrzeug_Marke"):
                    labels[i] = "B-Hersteller"

            # "X mm" uncommented
            if i < len(tokens) - 1 and tokens[i + 1].lower() in ["mm", "cm"]:
                if tokens[i].isdigit() and labels[i] == "O":
                    labels[i] = "B-Durchmesser"

        return labels

    def _resolve_tag_with_context(
        self, original_tag: str, position: int, tokens: List[str], entity: str
    ) -> str:
        """
        Disambiguate tags using context, mainly between Marke and Hersteller.
        """
        if original_tag not in ["Kompatible_Fahrzeug_Marke", "Hersteller"]:
            return original_tag

        if entity.lower() not in self.known_brands:
            return original_tag

        # Look at previous token
        if position > 0:
            prev = tokens[position - 1].lower().strip(".,;:")
            if prev in ["für", "passend", "geeignet", "kompatibel", "fits", "fit"]:
                return "Kompatible_Fahrzeug_Marke"
            if prev in ["original", "oem", "genuine", "von"]:
                return "Hersteller"

        # Look at next token
        if position < len(tokens) - 1:
            nxt = tokens[position + 1].lower().strip(".,;:")
            if nxt in ["qualität", "original", "teil", "hersteller"]:
                return "Hersteller"

        # Default bias, i wonder if this seriously hinders some results.
        # TODO: Look into this
        return "Kompatible_Fahrzeug_Marke"

    def _fix_compound_entities(self, tokens: List[str], labels: List[str]) -> List[str]:
        fixed = labels.copy()
        for i in range(len(tokens) - 1):
            current = tokens[i]
            nxt = tokens[i + 1].lower()
            if current.isdigit() and nxt in ["mm", "cm", "m", "kg", "g", "stk", "x"]:
                if labels[i].startswith("B-") and labels[i + 1] == "O":
                    tag = labels[i][2:]
                    fixed[i + 1] = f"I-{tag}"
        return fixed


def build_gazetteers(tagged_df: pd.DataFrame) -> Dict[str, set]:
    """taking out known entities (lowercased) from training data per tag."""
    gazetteers = {}
    for tag in tagged_df["Tag"].unique():
        if tag and tag != "O":
            values = set(
                tagged_df[tagged_df["Tag"] == tag]["Token"]
                .str.lower()
                .str.strip()
                .unique()
            )
            gazetteers[tag] = {v for v in values if len(v) > 0}
    return gazetteers


def label_single_example(row_data, gazetteers, label2id) -> Dict[str, Any] | None:
    """
    Used inside multiprocessing: given (idx, text, category),
    returns HF-style dict or None (if no entities).
    """
    idx, text, category = row_data
    text = str(text or "").strip()
    if not text:
        return None

    labeler = ImprovedWeakNERLabeler(gazetteers, category)
    try:
        token_label_pairs = labeler.label_text(text)
    except Exception:
        return None

    tokens = [t for t, _ in token_label_pairs]
    labels = [lab for _, lab in token_label_pairs]

    label_ids = [label2id.get(lab, label2id["O"]) for lab in labels]

    if all(lid == label2id["O"] for lid in label_ids):
        return None

    return {
        "tokens": tokens,
        "ner_tags": label_ids,
        "Category": int(category),
    }


def create_weak_ner_dataset_parallel(
    unlabeled_df: pd.DataFrame,
    gazetteers: Dict[str, set],
    label2id: Dict[str, int],
    sample_size: int = 150_000,
    n_workers: int | None = None,
) -> Dataset:
    """
    Parallel-ized weak labeling over a sample of the unlabeled titles.
    """
    df = unlabeled_df.sample(n=min(sample_size, len(unlabeled_df)), random_state=42)

    row_data = [
        (idx, row.get("Title", ""), int(row.get("Category", 1)))
        for idx, row in df.iterrows()
    ]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)

    print(f"Generating weak labels using {n_workers} workers...")

    label_func = partial(label_single_example, gazetteers=gazetteers, label2id=label2id)

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(label_func, row_data, chunksize=100),
                total=len(row_data),
                desc="Weak labeling (improved)",
            )
        )

    weak_examples = [r for r in results if r is not None]
    skipped = len(results) - len(weak_examples)
    print(f"created {len(weak_examples)} weakly-labeled examples")
    print(f"skipped {skipped} examples")
    print(f"overall coverage: {len(weak_examples) / len(df) * 100:.1f}%")

    return Dataset.from_list(weak_examples)


def load_tagged_and_unlabeled():
    tagged = pd.read_csv(
        DATA_DIR / "Tagged_Titles_Train.tsv",
        sep="\t",
        keep_default_na=False,
        na_values=None,
    )

    unlabeled = pd.read_csv(
        DATA_DIR / "Listing_Titles.tsv",
        sep="\t",
        keep_default_na=False,
        na_values=None,
    )

    label_list, label2id, id2label = build_bio_labels(tagged)
    gazetteers = build_gazetteers(tagged)

    return tagged, unlabeled, (label_list, label2id, id2label), gazetteers
