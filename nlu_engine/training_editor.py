"""
Training Editor Module for NLU Model Management

Provides functions to:
- Load/save intents from intents.json
- Add/delete training examples
- Retrain the intent classification model
- Handle model updates without app restart
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # BANKBOT_AI
INTENTS_PATH = BASE_DIR / "nlu_engine" / "intents.json"
MODEL_DIR = BASE_DIR / "models" / "intent_model"


def _normalize_examples_list(examples: List[str]) -> List[str]:
    """Normalize a list of examples.

    - Split any items that contain multiple lines
    - Strip whitespace
    - Remove leading numeric prefixes (e.g. "21. ")
    - Remove duplicates while preserving order
    - Remove stray backslashes and empty entries
    """
    import re

    normalized: List[str] = []
    seen = set()

    for item in examples:
        if item is None:
            continue
        # Ensure string
        text = str(item)
        # Split lines if multiple examples were pasted together
        parts = text.splitlines()

        for part in parts:
            p = part.strip()
            if not p:
                continue
            # Remove leading numbered prefixes like "21. " or repeated "22. 21. "
            p = re.sub(r'^\s*(?:\d+\.\s*)+', '', p)
            # Remove stray backslashes or control characters left in paste
            p = p.replace('\\', '').strip()
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            normalized.append(p)

    return normalized


def load_intents() -> List[Dict[str, Any]]:
    """Load intents from intents.json.
    
    Returns:
        List of intent dictionaries with 'name' and 'examples' keys.
        Returns empty list if file doesn't exist.
    """
    try:
        if not INTENTS_PATH.exists():
            return []

        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        intents = data.get("intents", [])
        # Normalize examples on load to avoid UI showing duplicated or
        # newline-combined entries. This does not overwrite the file here.
        for intent in intents:
            intent_examples = intent.get("examples", []) or []
            intent["examples"] = _normalize_examples_list(intent_examples)

        return intents
    except Exception as e:
        print(f"Error loading intents: {e}")
        return []


def save_intents(intents: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Save intents back to intents.json.
    
    Args:
        intents: List of intent dictionaries
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Ensure directory exists
        INTENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean intents before saving to ensure examples are normalized
        cleaned = []
        for intent in intents:
            examples = intent.get("examples", []) or []
            cleaned_examples = _normalize_examples_list(examples)
            cleaned.append({"name": intent.get("name"), "examples": cleaned_examples})

        data = {"intents": cleaned}
        with open(INTENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True, f"✅ Saved {len(intents)} intents to intents.json"
    except Exception as e:
        return False, f"❌ Error saving intents: {str(e)}"


def add_example(intent_name: str, example: str) -> Tuple[bool, str]:
    """Add a training example to an intent.
    
    Args:
        intent_name: Name of the intent
        example: New training example text
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not example or not str(example).strip():
        return False, "Example text cannot be empty"

    # Allow multiple examples separated by newlines
    new_examples = _normalize_examples_list([example])

    intents = load_intents()

    # Find intent and append new unique examples
    for intent in intents:
        if intent["name"] == intent_name:
            added = 0
            for ex in new_examples:
                if ex not in intent["examples"]:
                    intent["examples"].append(ex)
                    added += 1

            if added == 0:
                return False, "⚠️  No new examples added (duplicates)"

            success, msg = save_intents(intents)
            if success:
                return True, f"✅ Added {added} example(s) to '{intent_name}'"
            return False, msg

    return False, f"Intent '{intent_name}' not found"


def delete_example(intent_name: str, example_index: int) -> Tuple[bool, str]:
    """Delete a training example from an intent.
    
    Args:
        intent_name: Name of the intent
        example_index: Index of example to delete (0-based)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    intents = load_intents()
    
    for intent in intents:
        if intent["name"] == intent_name:
            if 0 <= example_index < len(intent["examples"]):
                deleted_text = intent["examples"].pop(example_index)
                success, msg = save_intents(intents)
                if success:
                    return True, f"✅ Deleted example: '{deleted_text}'"
                return False, msg
            else:
                return False, "Invalid example index"
    
    return False, f"Intent '{intent_name}' not found"


def create_intent(intent_name: str, examples: List[str] = None) -> Tuple[bool, str]:
    """Create a new intent.
    
    Args:
        intent_name: Name of the new intent
        examples: Optional list of initial examples
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not intent_name or not intent_name.strip():
        return False, "Intent name cannot be empty"
    
    intent_name = intent_name.strip().lower().replace(" ", "_")
    
    intents = load_intents()
    
    # Check if intent already exists
    for intent in intents:
        if intent["name"] == intent_name:
            return False, f"Intent '{intent_name}' already exists"
    
    # Normalize examples: accept either a list of strings or a single
    # newline-separated string from the UI and convert to a clean list.
    normalized_examples: List[str] = []
    if examples:
        if isinstance(examples, str):
            # split on any common newline and strip each line
            normalized_examples = [e.strip() for e in examples.splitlines() if e.strip()]
        elif isinstance(examples, list):
            normalized_examples = [str(e).strip() for e in examples if str(e).strip()]
        else:
            # fallback: try to coerce to string and split
            normalized_examples = [str(examples).strip()]

    new_intent = {
        "name": intent_name,
        "examples": normalized_examples
    }
    
    intents.append(new_intent)
    success, msg = save_intents(intents)
    
    if success:
        return True, f"✅ Created intent '{intent_name}'"
    return False, msg


def delete_intent(intent_name: str) -> Tuple[bool, str]:
    """Delete an entire intent.
    
    Args:
        intent_name: Name of intent to delete
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    intents = load_intents()
    original_count = len(intents)
    
    intents = [i for i in intents if i["name"] != intent_name]
    
    if len(intents) == original_count:
        return False, f"Intent '{intent_name}' not found"
    
    success, msg = save_intents(intents)
    if success:
        return True, f"✅ Deleted intent '{intent_name}'"
    return False, msg


def clear_examples(intent_name: str) -> Tuple[bool, str]:
    """Remove all examples for a given intent and save file.

    Returns (success, message).
    """
    intents = load_intents()
    for intent in intents:
        if intent.get("name") == intent_name:
            intent["examples"] = []
            success, msg = save_intents(intents)
            if success:
                return True, f"✅ Cleared all examples for '{intent_name}'"
            return False, msg
    return False, f"Intent '{intent_name}' not found"


def build_training_data(intents: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, Dict]]]:
    """Build spaCy training data format from intents.
    
    Args:
        intents: List of intent dictionaries
        
    Returns:
        Tuple of (labels: List[str], train_data: List[Tuple[str, Dict]])
    """
    labels = [i["name"] for i in intents]
    train_data = []
    
    for intent in intents:
        name = intent["name"]
        for ex in intent.get("examples", []):
            cats = {label: 0.0 for label in labels}
            cats[name] = 1.0
            train_data.append((ex, {"cats": cats}))
    
    return labels, train_data


def retrain_model(
    n_iter: int = 20,
    batch_size: int = 8,
    drop: float = 0.2,
    learning_rate: float = 0.001,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Retrain the intent classification model.
    
    Loads intents.json, trains the model, and saves it to the model directory.
    
    Args:
        n_iter: Number of training iterations (epochs)
        batch_size: Batch size for training
        drop: Dropout rate
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of:
        - success: bool
        - message: str (status message)
        - stats: Dict with training stats (epochs, examples, labels)
    """
    try:
        intents = load_intents()
        
        if not intents:
            return False, "❌ No intents found in intents.json", {}
        
        labels, train_data = build_training_data(intents)
        
        if not train_data:
            return False, "❌ No training examples found", {}
        
        # Create blank English model
        nlp = spacy.blank("en")
        
        # Add text categorizer
        if "textcat" in nlp.pipe_names:
            nlp.remove_pipe("textcat")
        
        textcat = nlp.add_pipe("textcat", last=True)
        
        # Add labels
        for label in labels:
            textcat.add_label(label)
        
        # Initialize the model with a dummy batch to set dimensions
        # This prevents "Cannot get dimension 'nO'" error
        dummy_examples = []
        for text, annot in train_data[:min(3, len(train_data))]:
            doc = nlp.make_doc(text)
            dummy_examples.append(Example.from_dict(doc, annot))
        
        # Initialize the pipeline
        nlp.initialize(lambda: dummy_examples)
        
        # Initialize optimizer with learning rate
        optimizer = nlp.create_optimizer()
        # Set learning rate for the optimizer
        optimizer.learn_rate = learning_rate
        
        # Training loop
        epoch_losses = []
        
        for epoch in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            # Create batches
            batches = list(minibatch(train_data, size=batch_size))
            
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                
                for text, annot in zip(texts, annotations):
                    doc = nlp.make_doc(text)
                    examples.append(Example.from_dict(doc, annot))
                
                nlp.update(
                    examples,
                    sgd=optimizer,
                    drop=drop,
                    losses=losses
                )
            
            epoch_loss = losses.get("textcat", 0)
            epoch_losses.append(epoch_loss)
        
        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(MODEL_DIR)
        
        stats = {
            "epochs": n_iter,
            "examples": len(train_data),
            "intents": len(labels),
            "batch_size": batch_size,
            "final_loss": float(epoch_losses[-1]) if epoch_losses else 0
        }
        
        message = (
            f"✅ Model trained successfully!\n"
            f"  • {len(labels)} intents\n"
            f"  • {len(train_data)} examples\n"
            f"  • {n_iter} epochs\n"
            f"  • Final loss: {stats['final_loss']:.4f}"
        )
        
        return True, message, stats
        
    except Exception as e:
        return False, f"❌ Training failed: {str(e)}", {}


def get_intent_stats() -> Dict[str, Any]:
    """Get statistics about current intents.
    
    Returns:
        Dictionary with intent statistics
    """
    intents = load_intents()
    
    stats = {
        "total_intents": len(intents),
        "total_examples": sum(len(i.get("examples", [])) for i in intents),
        "by_intent": {}
    }
    
    for intent in intents:
        stats["by_intent"][intent["name"]] = {
            "count": len(intent.get("examples", [])),
            "examples": intent.get("examples", [])
        }
    
    return stats


def validate_intent_data() -> Tuple[bool, List[str]]:
    """Validate intent data and return list of issues.
    
    Returns:
        Tuple of (is_valid: bool, issues: List[str])
    """
    intents = load_intents()
    issues = []
    
    if not intents:
        issues.append("No intents found in intents.json")
    
    for intent in intents:
        name = intent.get("name")
        examples = intent.get("examples", [])
        
        if not name:
            issues.append("Found intent with no name")
        
        if len(examples) < 5:
            issues.append(f"Intent '{name}' has only {len(examples)} examples (recommend 5+)")
        
        # Check for empty examples
        for i, ex in enumerate(examples):
            if not ex or not ex.strip():
                issues.append(f"Intent '{name}' has empty example at index {i}")
    
    return len(issues) == 0, issues


# ============================================================================
# MODEL RELOADING & NLU INFERENCE (for Streamlit UI integration)
# ============================================================================

def reload_intent_model():
    """Force reload the intent model from disk.
    
    This is used after retraining to update the running model without app restart.
    Works by invalidating the global cache in infer_intent module.
    """
    try:
        # Import and reset the global cache
        from nlu_engine import infer_intent
        infer_intent.nlp_intent = None  # Clear cached model
        # Force reload on next use
        model = infer_intent._load_intent_model()
        return True, "✅ Model reloaded successfully"
    except Exception as e:
        return False, f"❌ Failed to reload model: {str(e)}"


def predict_intent_with_confidence(query: str) -> Dict[str, Any]:
    """Predict intent and confidence for a user query.
    
    Args:
        query: User input text
        
    Returns:
        Dictionary with:
        - success: bool
        - intent: str (predicted intent name)
        - confidence: float (0-1)
        - all_scores: Dict[str, float] (all intent scores)
        - error: str (if failed)
    """
    try:
        if not query or not query.strip():
            return {
                "success": False,
                "intent": None,
                "confidence": 0.0,
                "all_scores": {},
                "error": "Query cannot be empty"
            }
        
        from nlu_engine.infer_intent import predict_intent
        
        # predict_intent returns list of (intent, score) tuples
        predictions = predict_intent(query)
        
        if not predictions:
            return {
                "success": False,
                "intent": None,
                "confidence": 0.0,
                "all_scores": {},
                "error": "No predictions generated"
            }
        
        # Top prediction
        top_intent, top_score = predictions[0]
        
        # Build all_scores dict
        all_scores = {intent: float(score) for intent, score in predictions}
        
        return {
            "success": True,
            "intent": top_intent,
            "confidence": float(top_score),
            "all_scores": all_scores,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "intent": None,
            "confidence": 0.0,
            "all_scores": {},
            "error": str(e)
        }


def extract_entities_from_query(query: str) -> Dict[str, Any]:
    """Extract entities from a user query.
    
    Args:
        query: User input text
        
    Returns:
        Dictionary with:
        - success: bool
        - entities: List[Dict] with entity data
        - error: str (if failed)
    """
    try:
        if not query or not query.strip():
            return {
                "success": True,
                "entities": [],
                "error": None
            }
        
        from nlu_engine.entity_extractor import extract_entities
        entities = extract_entities(query)
        
        return {
            "success": True,
            "entities": entities if entities else [],
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "entities": [],
            "error": str(e)
        }


def get_model_info() -> Dict[str, Any]:
    """Get information about the current model.
    
    Returns:
        Dictionary with model metadata
    """
    info = {
        "model_exists": MODEL_DIR.exists(),
        "model_path": str(MODEL_DIR),
        "intents_path": str(INTENTS_PATH),
        "intents": load_intents(),
    }
    
    if MODEL_DIR.exists():
        try:
            # Check model file size
            model_size = sum(
                f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()
            ) / (1024 * 1024)  # Convert to MB
            info["model_size_mb"] = round(model_size, 2)
            info["model_status"] = "✅ Ready"
        except Exception:
            info["model_status"] = "⚠️  Cannot read model"
    else:
        info["model_status"] = "❌ Not found - please train model"
    
    return info
