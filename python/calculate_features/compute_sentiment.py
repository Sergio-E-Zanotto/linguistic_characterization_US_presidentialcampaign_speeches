#!/usr/bin/env python3.10
"""
Compute sentiment for paragraphs using a pretrained RoBERTa sentiment model
(non-Twitter-specific), and write results to CSV.

Model:
- siebert/sentiment-roberta-large-english  (binary: POSITIVE / NEGATIVE)

Input:
- prof_llm_sentiment.csv   (must contain column: 'text')

Output:
- prof_llm_sentiment_with_roberta.csv
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# -----------------------------
# CONFIG
# -----------------------------
TEXT_CSV = "prof_llm_sentiment.csv"
TEXT_COLUMN = "text"

MODEL_NAME = "siebert/sentiment-roberta-large-english"
OUTPUT_CSV = "prof_llm_sentiment_with_roberta.csv"

BATCH_SIZE = 64
MAX_LENGTH = 256  # set 512 if your paragraphs are long (slower)

# -----------------------------
# HELPERS
# -----------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

@torch.no_grad()
def predict_proba(texts, tokenizer, model, device, batch_size=64, max_length=256):
    """
    Returns probabilities with shape (n, 2).
    For this model the labels are typically: NEGATIVE / POSITIVE.
    We'll detect label mapping from model.config.id2label when available.
    """
    probs_all = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch = texts[start:start + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs = softmax(logits)
        probs_all.append(probs)

    return np.vstack(probs_all)

def get_label_order(model) -> list[str]:
    """
    Try to recover label order from id2label (0..K-1).
    Fallback to ["NEGATIVE","POSITIVE"].
    """
    try:
        id2label = model.config.id2label
        # ensure sorted by id
        return [id2label[i] for i in sorted(id2label.keys())]
    except Exception:
        return ["NEGATIVE", "POSITIVE"]

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading data...")
    df = pd.read_csv(TEXT_CSV, low_memory=False)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in {TEXT_CSV}")

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    print(f"Paragraphs: {len(texts)}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    labels = get_label_order(model)
    print("Label order:", labels)

    probs = predict_proba(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # Map probabilities to named columns robustly
    # We normalize names to upper-case for matching.
    labels_up = [l.upper() for l in labels]
    if "NEGATIVE" in labels_up and "POSITIVE" in labels_up:
        neg_idx = labels_up.index("NEGATIVE")
        pos_idx = labels_up.index("POSITIVE")
    else:
        # fallback: assume [NEG, POS]
        neg_idx, pos_idx = 0, 1

    df["rob_neg"] = probs[:, neg_idx].astype(np.float32)
    df["rob_pos"] = probs[:, pos_idx].astype(np.float32)

    # Predicted label
    pred_idx = np.argmax(probs, axis=1)
    df["rob_label"] = [labels[i] for i in pred_idx]

    # Two useful derived measures (similar spirit to VADER)
    df["rob_compound"] = (df["rob_pos"] - df["rob_neg"]).astype(np.float32)  # [-1, 1]
    df["rob_intensity"] = (
    df["rob_pos"] + df["rob_neg"]
).astype(np.float32)


    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    print("Wrote columns: rob_neg, rob_pos, rob_label, rob_compound, rob_intensity")

if __name__ == "__main__":
    main()
