#!/usr/bin/env python3.10
"""
Compute paragraph-level formality using a RoBERTa model.

Model:
- s-nlp/roberta-base-formality
  Output is binary (formal vs informal), but we use
  P(formal) as a continuous formality score in [0, 1].

"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# -----------------------------
# CONFIG
# -----------------------------

INPUT_CSV = "prof_llm_formality.csv"
OUTPUT_CSV = "llm_formality.csv"

TEXT_COLUMN = "text"

MODEL_NAME = "s-nlp/roberta-base-formality-ranker"

BATCH_SIZE = 16        # safe for CPU / modest GPU
MAX_LENGTH = 512       # RoBERTa max length


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("Loading data...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in {INPUT_CSV}")

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"Paragraphs: {n}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    formality_scores = np.zeros(n, dtype=np.float32)

    print("Computing formality scores...")
    for start in tqdm(range(0, n, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n)
        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)

        # LABEL_1 = formal → use P(formal)
        formality_scores[start:end] = probs[:, 1].cpu().numpy()

    # Store results
    df["formality"] = formality_scores
    df["informality"] = 1.0 - df["formality"]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    print("Wrote columns: formality, informality")


if __name__ == "__main__":
    main()
