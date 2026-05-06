#!/usr/bin/env python3.10
"""
Compute semantic similarity between paragraphs and a vulgarity/aggression lexicon
using sentence-transformer embeddings.

Inputs:
- prof_llm.csv               (must contain column: 'text')
- vulgarity_vocabulary.csv   (must contain column: 'term')

Output:
- prof_llm_with_similarity.csv
"""

import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

TEXT_CSV = "prof_llm.csv"
LEXICON_CSV = "vulgarity_vocabulary.csv"
TEXT_COLUMN = "text"
LEXICON_COLUMN = "term"

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_CSV = "prof_llm_with_similarity.csv"

BATCH_SIZE_TEXT = 64
BATCH_SIZE_LEX = 256

# Top-k settings
TOPK = 10  # mean of top-10 term similarities
WRITE_MAX = True  # also write max similarity


# -----------------------------
# MAIN
# -----------------------------

def main():
    print("Loading data...")
    df = pd.read_csv(TEXT_CSV, low_memory=False)
    lex = pd.read_csv(LEXICON_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in {TEXT_CSV}")

    if LEXICON_COLUMN not in lex.columns:
        raise ValueError(f"Column '{LEXICON_COLUMN}' not found in {LEXICON_CSV}")

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    terms = (
        lex[LEXICON_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    print(f"Paragraphs: {len(texts)}")
    print(f"Lexicon terms: {len(terms)}")

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding lexicon...")
    term_emb = model.encode(
        terms,
        batch_size=BATCH_SIZE_LEX,
        normalize_embeddings=True,
        show_progress_bar=True,
    )  # shape: (n_terms, d)

    print("Embedding paragraphs...")
    para_emb = model.encode(
        texts,
        batch_size=BATCH_SIZE_TEXT,
        normalize_embeddings=True,
        show_progress_bar=True,
    )  # shape: (n_paras, d)

    print("Computing top-k cosine similarity...")
    # Since embeddings are normalized, cosine similarity = dot product.
    # Compute in blocks to avoid building a huge dense matrix if data grows.
    n_paras = para_emb.shape[0]
    n_terms = term_emb.shape[0]

    topk_mean = np.empty(n_paras, dtype=np.float32)
    topk_max = np.empty(n_paras, dtype=np.float32) if WRITE_MAX else None

    # Blocked multiplication: (block, d) @ (d, n_terms) -> (block, n_terms)
    block_size = 2048
    term_T = term_emb.T  # (d, n_terms)

    for start in range(0, n_paras, block_size):
        end = min(start + block_size, n_paras)
        S = para_emb[start:end] @ term_T  # (block, n_terms)

        # top-k mean
        k = min(TOPK, n_terms)
        # partition is faster than full sort
        topk_vals = np.partition(S, -k, axis=1)[:, -k:]
        topk_mean[start:end] = topk_vals.mean(axis=1)

        if WRITE_MAX:
            topk_max[start:end] = S.max(axis=1)

    df[f"agg_sim_top{TOPK}"] = topk_mean
    if WRITE_MAX:
        df["agg_sim_max"] = topk_max

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    print(f"Wrote columns: agg_sim_top{TOPK}" + (", agg_sim_max" if WRITE_MAX else ""))


if __name__ == "__main__":
    main()

