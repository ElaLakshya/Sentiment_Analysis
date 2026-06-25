"""
fasttext_client.py

FastText taxonomy classification — but only when a model trained on
tweet/news data exists at MODEL_PATH.

The bundled taxonomy_model.bin is trained on e-commerce product data.
It has no concept of politics, philosophy, economics, or current events.
Feeding it those topics produces nonsense labels (e.g. "Ski Goggles" for
a Trump tweet).

Current strategy:
  - If taxonomy_model.bin exists AND was trained on non-product data → use it.
  - Otherwise → return None, and the caller (ultimate_pipeline.py) falls
    back to BitNet which handles all content types correctly.

To train a proper model later:
  1. Export your collected tweets with their correct labels to a .txt file:
       __label__Politics  Trump signed the bill today...
       __label__Technology  OpenAI released GPT-5...
  2. Run:  python train_taxonomy.py
  3. Drop the resulting taxonomy_model.bin here and set USE_FASTTEXT = True.
"""

import os
import fasttext

MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH", "taxonomy_model.bin")

# ── Safety switch ──────────────────────────────────────────────────────────────
# Set to True ONLY after you have retrained the model on tweet/news content.
# Leaving this False routes taxonomy through BitNet, which works correctly
# on any content type out of the box.
USE_FASTTEXT = True
# ──────────────────────────────────────────────────────────────────────────────

_model = None


def get_model():
    global _model
    if not USE_FASTTEXT:
        return None
    if _model is None:
        if os.path.exists(MODEL_PATH):
            fasttext.FastText.eprint = lambda x: None
            _model = fasttext.load_model(MODEL_PATH)
        else:
            print(f"[fasttext] Model not found at {MODEL_PATH}. Falling back to BitNet.")
            return None
    return _model


def classify_taxonomy(text: str, k: int = 3, threshold: float = 0.05):
    """
    Returns a list of taxonomy dicts, or None if FastText is disabled/unavailable.
    Returning None signals the caller to use BitNet for taxonomy instead.
    """
    model = get_model()
    if not model:
        return None     # ← caller should use BitNet taxonomy

    clean_text = text.replace("\n", " ").replace("\r", " ").strip()
    if not clean_text:
        return []

    labels, probabilities = model.predict(clean_text, k=k)

    results = []
    for label, prob in zip(labels, probabilities):
        if prob >= threshold:
            clean_label = label.replace("__label__", "").replace("_", " ")
            results.append({"top_level": clean_label, "sub_level": None})

    if not results:
        results.append({"top_level": "Uncategorized", "sub_level": None})

    return results
