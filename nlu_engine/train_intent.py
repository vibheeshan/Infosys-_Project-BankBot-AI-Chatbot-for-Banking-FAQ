import json
import random
from pathlib import Path
from datetime import datetime

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # BANKBOT_AI
INTENTS_PATH = BASE_DIR / "nlu_engine" / "intents.json"
MODEL_DIR = BASE_DIR / "models" / "intent_model"
LOG_PATH = BASE_DIR / "models" / "training.log"


def load_intents():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def build_training_data(intents):
    """
    spaCy textcat format:
    (text, {"cats": {"intent_name": 1 or 0, ...}})
    """
    labels = [i["name"] for i in intents]
    train_data = []

    for intent in intents:
        name = intent["name"]
        for ex in intent["examples"]:
            cats = {label: 0.0 for label in labels}
            cats[name] = 1.0
            train_data.append((ex, {"cats": cats}))

    return labels, train_data


def train_intent_model(
    n_iter: int = 20,
    batch_size: int = 8,
    drop: float = 0.2,
    learning_rate: float | None = None,
):
    intents = load_intents()
    labels, train_data = build_training_data(intents)

    print(f"Loaded {len(intents)} intents, {len(train_data)} training examples.")

    # blank English pipeline
    nlp = spacy.blank("en")

    # add text categorizer (use default config to be compatible with installed spaCy)
    textcat = nlp.add_pipe("textcat")

    # add labels
    for label in labels:
        textcat.add_label(label)

    # start training
    optimizer = nlp.begin_training()

    # try to set a fixed learning rate if provided
    detected_lr = None
    if learning_rate is not None:
        try:
            if hasattr(optimizer, "alpha"):
                optimizer.alpha = learning_rate
                detected_lr = learning_rate
            elif hasattr(optimizer, "learn_rate"):
                optimizer.learn_rate = learning_rate
                detected_lr = learning_rate
        except Exception:
            detected_lr = None

    epoch_losses = []

    for it in range(n_iter):
        random.shuffle(train_data)
        losses = {}

        # use fixed batch size
        batches = list(minibatch(train_data, size=batch_size))
        avg_batch = batch_size if batches else 0

        for batch in batches:
            texts, annotations = zip(*batch)
            examples = []
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                examples.append(Example.from_dict(doc, annotations[i]))
            nlp.update(examples, sgd=optimizer, drop=drop, losses=losses)

        epoch_losses.append(losses.get("textcat", losses))

        # try to read a learning-rate value from the optimizer for reporting
        current_lr = detected_lr
        try:
            if current_lr is None:
                if hasattr(optimizer, "alpha"):
                    current_lr = getattr(optimizer, "alpha")
                elif hasattr(optimizer, "learn_rate"):
                    current_lr = getattr(optimizer, "learn_rate")
        except Exception:
            current_lr = detected_lr

        print(
            f"Iteration {it + 1}/{n_iter} - Loss: {losses} - batch_size: {avg_batch} - lr: {current_lr}"
        )

    # save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"Saved intent model to {MODEL_DIR}")

    # log summary
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"{datetime.now().isoformat()} - trained {len(intents)} intents, "
            f"{len(train_data)} examples, {n_iter} epochs, batch_size={batch_size}, drop={drop}, lr={detected_lr}\n"
        )

    # write training metadata into the model dir
    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_intents": len(intents),
        "n_examples": len(train_data),
        "n_iter": n_iter,
        "batch_size": batch_size,
        "dropout": drop,
        "learning_rate": detected_lr,
        "epoch_losses": epoch_losses,
    }

    try:
        with open(MODEL_DIR / "training_meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
    except Exception:
        pass

    return str(MODEL_DIR)


if __name__ == "__main__":
    train_intent_model()
