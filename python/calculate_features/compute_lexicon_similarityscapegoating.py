#!/usr/bin/env python3.10
"""
Compute semantic similarity between paragraphs and a scapegoating lexicon
using sentence-transformer embeddings (top-k aggregation).

Inputs:
- prof_llm_scapegoating.csv     (must contain column: 'text')
- scapegoating_vocabulary.csv   (should contain columns: 'term','category'
  BUT this script is robust if your file is malformed and 'term,category'
  ended up in a single column; it will split on the first comma.)

Output:
- prof_llm_scapegoating_with_similarity.csv
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
TEXT_CSV = "prof_llm_scapegoating.csv"
LEXICON_CSV = "scapegoating_vocabulary.csv"
TEXT_COLUMN = "text"

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_CSV = "prof_llm_scapegoating_with_similarity.csv"

BATCH_SIZE_TEXT = 64
BATCH_SIZE_LEX = 256

TOPK = 10
WRITE_MAX = True
BLOCK_SIZE = 2048


# -----------------------------
# HELPERS
# -----------------------------
def load_scapegoating_terms(path: str) -> list[str]:
    """
    Loads scapegoating terms robustly:
    - If CSV has a proper 'term' column, use it.
    - If CSV was read as one column (e.g., header like 'term\tcategory'
      and rows like 'fault of,blame'), split each row at the first comma.
    """
    lex_raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    print("Raw dataframe shape:", lex_raw.shape)
    print("Raw columns:", lex_raw.columns.tolist())
    print("Raw head:")
    print(lex_raw.head())
    # Strip BOM/whitespace from column names
    lex_raw.columns = [c.strip().replace("\ufeff", "") for c in lex_raw.columns]

    if "term" in lex_raw.columns:
        term_series = lex_raw["term"]
    elif lex_raw.shape[1] == 1:
        # One-column malformed case: split row values on first comma
        only_col = lex_raw.columns[0]
        term_series = (
            lex_raw[only_col]
            .astype(str)
            .str.split(",", n=1, expand=True)[0]
        )
    else:
        # Fallback: first column
        term_series = lex_raw.iloc[:, 0]

    terms = (
        term_series
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"": np.nan, "term": np.nan})
        .dropna()
        .unique()
        .tolist()
    )

    if len(terms) == 0:
        raise ValueError(f"No terms could be loaded from {path}. Check file format.")
    return terms


def topk_scores(para_emb: np.ndarray, term_emb: np.ndarray, k: int, write_max: bool):
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

    print("Loading scapegoating lexicon...")
    terms = load_scapegoating_terms(LEXICON_CSV)
    print(f"Lexicon terms: {len(terms)}")

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding lexicon...")
    term_emb = model.encode(
        terms,
        batch_size=BATCH_SIZE_LEX,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Embedding paragraphs...")
    para_emb = model.encode(
        texts,
        batch_size=BATCH_SIZE_TEXT,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Computing top-{TOPK} cosine similarity...")
    topk_mean, topk_max = topk_scores(para_emb, term_emb, TOPK, WRITE_MAX)

    df[f"scapegoating_sim_top{TOPK}"] = topk_mean
    if WRITE_MAX:
        df["scapegoating_sim_max"] = topk_max

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    print(
        f"Wrote columns: scapegoating_sim_top{TOPK}"
        + (", scapegoating_sim_max" if WRITE_MAX else "")
    )


if __name__ == "__main__":
    main()
