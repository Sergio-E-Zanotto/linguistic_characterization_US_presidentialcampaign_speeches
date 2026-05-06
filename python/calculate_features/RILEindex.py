#!/usr/bin/env python3.10
"""
Compute RILE-style score per text row using:
manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1

Input:
- prof_llm_crisis.csv (must contain column: 'text')

Output:
- prof_llm_crisis_with_rile.csv
Adds:
  left_mass, right_mass, rile_raw, rile_norm, top_label, top_prob
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------
TEXT_CSV = "prof_llm_crisis.csv"
TEXT_COLUMN = "text"

MODEL_ID = "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1"
TOKENIZER_ID = "xlm-roberta-large"  # IMPORTANT: tokenizer lives here, not in the model repo

OUTPUT_CSV = "prof_llm_crisis_with_rile.csv"

BATCH_SIZE = 16          # pairs + max_length padding can be heavier; 16 is safe
MAX_LENGTH = 300         # authors say they fine-tuned with 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Soft probabilities recommended for continuous ideology scoring
USE_SOFT_PROBS = True

# -----------------------------
# RILE MAPPING (strings must match model.config.id2label exactly)
# -----------------------------
LEFT_LABELS = {
    "103 - Anti-Imperialism",
    "105 - Military: Negative",
    "106 - Peace",
    "107 - Internationalism: Positive",
    "202 - Democracy",
    "403 - Market Regulation",
    "404 - Economic Planning",
    "406 - Protectionism: Positive",
    "412 - Controlled Economy",
    "413 - Nationalisation",
    "504 - Welfare State Expansion",
    "506 - Education Expansion",
    "701 - Labour Groups: Positive",
}

RIGHT_LABELS = {
    "104 - Military: Positive",
    "201 - Freedom and Human Rights",
    "203 - Constitutionalism: Positive",
    "305 - Political Authority",
    "401 - Free Market Economy",
    "402 - Incentives",
    "407 - Protectionism: Negative",
    "414 - Economic Orthodoxy",
    "505 - Welfare State Limitation",
    "601 - National Way of Life: Positive",
    "603 - Traditional Morality: Positive",
    "605 - Law and Order: Positive",
    "606 - Civic Mindedness: Positive",
}

# -----------------------------
# HELPERS
# -----------------------------
def batched(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i : i + n]

def main():
    print("Loading data...")
    df = pd.read_csv(TEXT_CSV, low_memory=False)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in {TEXT_CSV}. Found: {df.columns.tolist()}")

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    print(f"Rows: {len(texts)}")

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Build label index mapping
    id2label = model.config.id2label
    label2id = {id2label[i]: i for i in range(len(id2label))}

    # Validate labels exist
    missing_left = sorted([l for l in LEFT_LABELS if l not in label2id])
    missing_right = sorted([l for l in RIGHT_LABELS if l not in label2id])
    if missing_left or missing_right:
        raise ValueError(
            "Some RILE labels were not found in model.config.id2label.\n"
            f"Missing left: {missing_left}\n"
            f"Missing right: {missing_right}\n"
            "Fix: print(model.config.id2label) and update LEFT_LABELS/RIGHT_LABELS exactly."
        )

    left_idx = np.array([label2id[l] for l in LEFT_LABELS], dtype=np.int64)
    right_idx = np.array([label2id[l] for l in RIGHT_LABELS], dtype=np.int64)

    n = len(texts)
    left_mass = np.empty(n, dtype=np.float32)
    right_mass = np.empty(n, dtype=np.float32)
    top_prob = np.empty(n, dtype=np.float32)
    top_label = [""] * n

    print(f"Scoring batches on {DEVICE}...")

    with torch.no_grad():
        for start, batch_texts in batched(texts, BATCH_SIZE):
            # For “one score per row”, we use the row text as BOTH:
            #   sentence = row_text
            #   context  = row_text
            # This matches authors’ guidance for no additional context.
            sentences = batch_texts
            contexts = batch_texts

            enc = tokenizer(
                sentences,
                contexts,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            logits = model(**enc).logits  # (batch, 56)

            if USE_SOFT_PROBS:
                probs = torch.softmax(logits, dim=1)  # (batch, 56)
                probs_np = probs.detach().cpu().numpy()

                left_mass[start : start + len(batch_texts)] = probs_np[:, left_idx].sum(axis=1)
                right_mass[start : start + len(batch_texts)] = probs_np[:, right_idx].sum(axis=1)

                tp = probs.max(dim=1)
                top_prob[start : start + len(batch_texts)] = tp.values.detach().cpu().numpy()
                top_ids = tp.indices.detach().cpu().numpy().tolist()
                for j, lid in enumerate(top_ids):
                    top_label[start + j] = id2label[lid]
            else:
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
                left_mass[start : start + len(batch_texts)] = np.isin(pred, left_idx).astype(np.float32)
                right_mass[start : start + len(batch_texts)] = np.isin(pred, right_idx).astype(np.float32)

                probs = torch.softmax(logits, dim=1)
                tp = probs.max(dim=1)
                top_prob[start : start + len(batch_texts)] = tp.values.detach().cpu().numpy()
                for j, lid in enumerate(pred.tolist()):
                    top_label[start + j] = id2label[lid]

    # RILE computations
    rile_raw = right_mass - left_mass
    denom = right_mass + left_mass
    rile_norm = np.where(denom > 0, rile_raw / denom, 0.0).astype(np.float32)

    df["left_mass"] = left_mass
    df["right_mass"] = right_mass
    df["rile_raw"] = rile_raw.astype(np.float32)
    df["rile_norm"] = rile_norm
    df["top_label"] = top_label
    df["top_prob"] = top_prob

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print("Added columns: left_mass, right_mass, rile_raw, rile_norm, top_label, top_prob")

if __name__ == "__main__":
    main()

