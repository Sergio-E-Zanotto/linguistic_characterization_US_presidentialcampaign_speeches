#!/usr/bin/env python3.10
"""
Compute semantic similarity between paragraphs and multiple lexicons
(victimism_us, victimism_me, victimism_general, scapegoating_them) using
sentence-transformer embeddings with top-k aggregation.

Inputs:
- prof_llm_scapegoating.csv   (must contain column: 'text')
- victimism_us.csv            (must contain column: 'term' OR be 1-col)
- victimism_me.csv            (must contain column: 'term' OR be 1-col)
- victimism_general.csv       (must contain column: 'term' OR be 1-col)
- scapegoating_them.csv       (must contain column: 'term' OR be 1-col)

Output:
- prof_llm_scapegoating_with_similarity_multi.csv
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
TEXT_CSV = "prof_llm_scapegoating.csv"
TEXT_COLUMN = "text"

# Lexicons to score (name -> csv path)
LEXICONS = {
    "victimism_us": "victimism_us.csv",
    "victimism_me": "victimism_me.csv",
    "victimism_general": "victimism_general.csv",
    "scapegoating_implicit": "scapegoating_implicit.csv",
    "scapegoating_them": "scapegoating_them.csv",
}

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_CSV = "prof_llm_scapegoating_with_similarity_multi.csv"

BATCH_SIZE_TEXT = 64
BATCH_SIZE_LEX = 256

TOPK = 10
WRITE_MAX = True
BLOCK_SIZE = 2048

# Optional: print some debug info for lexicons
DEBUG_LEXICON_LOAD = True
DEBUG_N_PREVIEW_TERMS = 10


# -----------------------------
# HELPERS
# -----------------------------
def load_terms_robust(path: str) -> list[str]:
    """
    Loads terms robustly from various possible CSV formats:
    - Proper 'term' column -> use it
    - Single-column CSV (including malformed header 'term,category') -> use column values;
      if values contain commas, split on first comma and keep left part
    - Fallback to first column

    Returns a unique, lowercased list of non-empty terms.
    """
    lex_raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Strip BOM/whitespace from column names
    lex_raw.columns = [c.strip().replace("\ufeff", "") for c in lex_raw.columns]

    if DEBUG_LEXICON_LOAD:
        print(f"\n[LEXICON LOAD] {path}")
        print("  Shape:", lex_raw.shape)
        print("  Columns:", lex_raw.columns.tolist())
        print("  Head:")
        print(lex_raw.head())

    if "term" in lex_raw.columns:
        term_series = lex_raw["term"]
    elif lex_raw.shape[1] == 1:
        # One-column case: values might be "fault of,blame" -> keep left part
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

    if DEBUG_LEXICON_LOAD:
        preview = terms[:DEBUG_N_PREVIEW_TERMS]
        print(f"  Loaded terms: {len(terms)} (preview: {preview})")

    return terms


def topk_scores(para_emb: np.ndarray, term_emb: np.ndarray, k: int, write_max: bool):
    """
    para_emb: (n_paras, d) normalized
    term_emb: (n_terms, d) normalized
    returns:
      mean_topk: (n_paras,)
      max_all: (n_paras,) if write_max
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

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Embed paragraphs once
    print("Embedding paragraphs...")
    para_emb = model.encode(
        texts,
        batch_size=BATCH_SIZE_TEXT,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # For each lexicon, embed terms and compute similarities
    for lex_name, lex_path in LEXICONS.items():
        print(f"\n=== Scoring lexicon: {lex_name} ({lex_path}) ===")
        terms = load_terms_robust(lex_path)
        print(f"Lexicon '{lex_name}' terms: {len(terms)}")

        print("Embedding lexicon terms...")
        term_emb = model.encode(
            terms,
            batch_size=BATCH_SIZE_LEX,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print(f"Computing top-{TOPK} cosine similarity for {lex_name}...")
        topk_mean, topk_max = topk_scores(para_emb, term_emb, TOPK, WRITE_MAX)

        df[f"{lex_name}_sim_top{TOPK}"] = topk_mean
        if WRITE_MAX:
            df[f"{lex_name}_sim_max"] = topk_max

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved output to: {OUTPUT_CSV}")

    written_cols = []
    for lex_name in LEXICONS.keys():
        written_cols.append(f"{lex_name}_sim_top{TOPK}")
        if WRITE_MAX:
            written_cols.append(f"{lex_name}_sim_max")
    print("Wrote columns:", ", ".join(written_cols))


if __name__ == "__main__":
    main()
