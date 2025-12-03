from typing import Dict, List, Tuple

import pandas as pd


def post_process_predictions(
    records: List[Tuple[int, int, str, str]],
    gazetteers: Dict[str, set],
    allow: Dict[int, set],
):
    """
    records: (Record Number, Category, Tag, Token)
    """
    corrected = []
    known_brands = {"bmw", "audi", "vw", "mercedes", "opel", "ford"}
    known_manufacturers = {"bosch", "ate", "brembo", "zimmermann", "febi"}
    position_words = {
        "va",
        "ha",
        "vorne",
        "hinten",
        "links",
        "rechts",
        "vl",
        "vr",
        "hl",
        "hr",
    }

    for rec, cat, tag, token in records:
        token_lower = token.lower().strip()

        # Gazetteer exclusivity
        if tag != "O":
            found_tags = [
                gaz_tag
                for gaz_tag, entities in gazetteers.items()
                if token_lower in entities and gaz_tag in allow[cat]
            ]
            if len(found_tags) == 1 and found_tags[0] != tag:
                tag = found_tags[0]

        # Brand/manufacturer disambiguation
        if token_lower in known_brands and tag == "Hersteller":
            tag = "Kompatible_Fahrzeug_Marke"
        elif token_lower in known_manufacturers and tag == "Kompatible_Fahrzeug_Marke":
            tag = "Hersteller"

        # Position indicators
        if token_lower in position_words and tag != "Einbauposition":
            tag = "Einbauposition"

        corrected.append((rec, cat, tag, token))

    return corrected
