from pathlib import Path
import spacy
from typing import List, Tuple
import json
from difflib import SequenceMatcher

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "intent_model"
INTENTS_PATH = BASE_DIR / "nlu_engine" / "intents.json"

# lazy-loaded spaCy pipeline for intent classification
nlp_intent = None


def _load_intent_model():
    """Load the intent model on first use and cache it.

    Raises RuntimeError with a helpful message if the model is not present
    or cannot be loaded.
    """
    global nlp_intent
    if nlp_intent is not None:
        return nlp_intent

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Intent model not found at {MODEL_DIR}. Please run `nlu_engine.train_intent.train_intent_model()` to create it."
        )

    try:
        nlp_intent = spacy.load(MODEL_DIR)
        return nlp_intent
    except Exception as e:
        raise RuntimeError(f"Failed to load intent model from {MODEL_DIR}: {e}") from e


def _load_intents_examples():
    try:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", [])
    except Exception:
        return []


def _fuzzy_intent_match(text: str, intents: List[dict], top_n: int = 3):
    """Fallback fuzzy matcher using examples in `intents.json`.

    Returns list of (intent_name, score) where score is similarity 0..1.
    """
    scores = {}
    for intent in intents:
        name = intent.get("name")
        examples = intent.get("examples", [])
        best = 0.0
        for ex in examples:
            r = SequenceMatcher(None, text.lower(), ex.lower()).ratio()
            if r > best:
                best = r
        scores[name] = best

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(k, float(v)) for k, v in sorted_scores[:top_n]]


def predict_intent(text: str, top_n: int = 3, min_score: float = 0.25) -> List[Tuple[str, float]]:
    """Predict intents for `text`.

    - Tries the trained spaCy model first.
    - If spaCy returns no categories or the top score is below `min_score`,
      falls back to fuzzy matching against intent examples.
    """
    nlp = _load_intent_model()
    doc = nlp(text)
    cats = getattr(doc, "cats", {}) or {}
    sorted_labels = sorted(cats.items(), key=lambda x: x[1], reverse=True)

    # If spaCy produced categories and top score is reasonably confident, return them
    if sorted_labels:
        top_score = sorted_labels[0][1]
        if top_score >= min_score:
            return sorted_labels[:top_n]

    # otherwise, fallback to fuzzy match using examples
    intents = _load_intents_examples()
    fuzzy = _fuzzy_intent_match(text, intents, top_n=top_n)

    # merge spaCy scores (if any) with fuzzy scores, preferring spaCy when above min_score
    merged = {}
    for k, v in fuzzy:
        merged[k] = max(merged.get(k, 0.0), v * 0.9)  # scale fuzzy slightly down
    for k, v in sorted_labels:
        merged[k] = max(merged.get(k, 0.0), float(v))

    merged_list = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return merged_list[:top_n]


if __name__ == "__main__":
    while True:
        msg = input("User: ")
        if msg.lower() in ("quit", "exit"):
            break
        print(predict_intent(msg))
