#!/usr/bin/env python3.10
"""
Compute sentiment for paragraphs using CardiffNLP Twitter-RoBERTa sentiment model
(3-class: NEGATIVE / NEUTRAL / POSITIVE) and write results to CSV.

Model:
- cardiffnlp/twitter-roberta-base-sentiment-latest  (3-class)

Input:
- prof_llm_sentiment.csv   (must contain column: 'text')

Output:
- prof_llm_sentiment_with_twitter_roberta.csv
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

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_CSV = "prof_llm_sentiment_with_twitter_roberta.csv"

BATCH_SIZE = 64
MAX_LENGTH = 256  # can set 512 if paragraphs are long (slower)

# Since you're NOT analyzing tweets, you can safely keep this False.
# If your text contains many @handles or URLs, setting True won't hurt.
APPLY_TWEET_PREPROCESSING = False

# -----------------------------
# HELPERS
# -----------------------------
def preprocess_twitter_style(text: str) -> str:
    """
    CardiffNLP recommends replacing usernames and URLs for tweet-like text.
    For speech paragraphs, this is optional. Keep off unless you have many
    @mentions / links.
    """
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

@torch.no_grad()
def predict_proba(texts, tokenizer, model, device, batch_size=64, max_length=256):
    """
    Returns probabilities with shape (n, K). Here K=3 for NEG/NEU/POS.
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
        probs = softmax_np(logits)
        probs_all.append(probs)

    return np.vstack(probs_all)

def get_label_order(model) -> list[str]:
    """
    Recover label names from id2label if available, otherwise assume
    the standard ordering used by the CardiffNLP model.
    """
    try:
        id2label = model.config.id2label
        return [id2label[i] for i in sorted(id2label.keys())]
    except Exception:
        # Common fallback order for this model
        return ["negative", "neutral", "positive"]

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

    if APPLY_TWEET_PREPROCESSING:
        texts = [preprocess_twitter_style(t) for t in texts]

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    labels = get_label_order(model)
    labels_up = [l.upper() for l in labels]
    print("Label order:", labels)

    probs = predict_proba(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # Robustly map indices
    def idx(label: str, fallback: int) -> int:
        return labels_up.index(label) if label in labels_up else fallback

    neg_idx = idx("NEGATIVE", 0)
    neu_idx = idx("NEUTRAL", 1)
    pos_idx = idx("POSITIVE", 2)

    df["tw_neg"] = probs[:, neg_idx].astype(np.float32)
    df["tw_neu"] = probs[:, neu_idx].astype(np.float32)
    df["tw_pos"] = probs[:, pos_idx].astype(np.float32)

    pred_idx = np.argmax(probs, axis=1)
    df["tw_label"] = [labels[i] for i in pred_idx]

    # Derived measures (useful for regression & comparison with VADER-like summaries)
    df["tw_compound"] = (df["tw_pos"] - df["tw_neg"]).astype(np.float32)   # [-1, 1]
    df["tw_intensity"] = (df["tw_pos"] + df["tw_neg"]).astype(np.float32) # [0, 1] (approx)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    print("Wrote columns: tw_neg, tw_neu, tw_pos, tw_label, tw_compound, tw_intensity")

if __name__ == "__main__":
    main()
