#!/usr/bin/env python3.10
"""
Compute semantic similarity between paragraphs and two lexicons ("us" and "them")
using sentence-transformer embeddings, with top-k aggregation.

Inputs:
- prof_llm_usthem.csv     (must contain column: 'text')
- us_vocabulary.csv       (must contain column: 'term')
- them_vocabulary.csv     (must contain column: 'term')

Output:
- prof_llm_usthem_with_similarity.csv
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

TEXT_CSV = "prof_llm_usthem.csv"
TEXT_COLUMN = "text"

US_LEXICON_CSV = "us_vocabulary.csv"
THEM_LEXICON_CSV = "them_vocabulary.csv"
LEXICON_COLUMN = "term"

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_CSV = "prof_llm_usthem_with_similarity.csv"

BATCH_SIZE_TEXT = 64
BATCH_SIZE_LEX = 256

# Top-k settings
TOPK = 10
WRITE_MAX = True

# Compute similarity matrix in blocks to manage memory
BLOCK_SIZE = 2048


# -----------------------------
# HELPERS
# -----------------------------

def load_terms(path: str, term_col: str = "term") -> list[str]:
    lex = pd.read_csv(path)
    if term_col not in lex.columns:
        raise ValueError(f"Column '{term_col}' not found in {path}")
    terms = (
        lex[term_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    if len(terms) == 0:
        raise ValueError(f"No terms loaded from {path}")
    return terms


def topk_scores(para_emb: np.ndarray, term_emb: np.ndarray, k: int, write_max: bool):
    """
    para_emb: (n_paras, d) normalized
    term_emb: (n_terms, d) normalized
    returns:
      topk_mean: (n_paras,)
      topk_max:  (n_paras,) or None
    """
    n_paras = para_emb.shape[0]
    n_terms = term_emb.shape[0]
    k = min(k, n_terms)

    out_mean = np.empty(n_paras, dtype=np.float32)
    out_max = np.empty(n_paras, dtype=np.float32) if write_max else None

    term_T = term_emb.T  # (d, n_terms)

    for start in range(0, n_paras, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_paras)
        S = para_emb[start:end] @ term_T  # (block, n_terms)

        topk_vals = np.partition(S, -k, axis=1)[:, -k:]
        out_mean[start:end] = topk_vals.mean(axis=1)

        if write_max:
            out_max[start:end] = S.max(axis=1)

    return out_mean, out_max


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

    print("Loading lexicons...")
    us_terms = load_terms(US_LEXICON_CSV, LEXICON_COLUMN)
    them_terms = load_terms(THEM_LEXICON_CSV, LEXICON_COLUMN)
    print(f"US terms: {len(us_terms)} | THEM terms: {len(them_terms)}")

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding paragraphs...")
    para_emb = model.encode(
        texts,
        batch_size=BATCH_SIZE_TEXT,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Embedding US lexicon...")
    us_emb = model.encode(
        us_terms,
        batch_size=BATCH_SIZE_LEX,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Embedding THEM lexicon...")
    them_emb = model.encode(
        them_terms,
        batch_size=BATCH_SIZE_LEX,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Computing top-{TOPK} similarity (US)...")
    us_topk, us_max = topk_scores(para_emb, us_emb, TOPK, WRITE_MAX)
    df[f"us_sim_top{TOPK}"] = us_topk
    if WRITE_MAX:
        df["us_sim_max"] = us_max

    print(f"Computing top-{TOPK} similarity (THEM)...")
    them_topk, them_max = topk_scores(para_emb, them_emb, TOPK, WRITE_MAX)
    df[f"them_sim_top{TOPK}"] = them_topk
    if WRITE_MAX:
        df["them_sim_max"] = them_max

    # Combined measures (top-k based)
    df[f"us_them_intensity_top{TOPK}"] = df[f"us_sim_top{TOPK}"] + df[f"them_sim_top{TOPK}"]
    df[f"us_them_diff_top{TOPK}"] = df[f"us_sim_top{TOPK}"] - df[f"them_sim_top{TOPK}"]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    cols_written = [
        f"us_sim_top{TOPK}",
        f"them_sim_top{TOPK}",
        f"us_them_intensity_top{TOPK}",
        f"us_them_diff_top{TOPK}",
    ]
    if WRITE_MAX:
        cols_written += ["us_sim_max", "them_sim_max"]
    print("Wrote columns:", ", ".join(cols_written))


if __name__ == "__main__":
    main()
