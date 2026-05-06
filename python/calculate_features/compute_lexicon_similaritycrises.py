#!/usr/bin/env python3.10
"""
Compute semantic similarity between paragraphs and a crisis lexicon
using sentence-transformer embeddings.

Scores produced:
- Overall (all terms)
- Per-category (each category treated as its own dictionary)

Inputs:
- prof_llm_crisis.csv        (must contain column: 'text')
- crisis_vocabulary.csv      (must contain columns: 'term', 'category')

Output:
- prof_llm_crisis_with_similarity.csv
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
TEXT_CSV = "prof_llm_crisis.csv"
LEXICON_CSV = "crisis_vocabulary.csv"

TEXT_COLUMN = "text"
LEXICON_COLUMN = "term"
CATEGORY_COLUMN = "category"

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_CSV = "prof_llm_crisis_with_similarity.csv"

BATCH_SIZE_TEXT = 64
BATCH_SIZE_LEX = 256

# Top-k settings
TOPK = 10               # mean of top-10 term similarities
WRITE_MAX = True        # also write max similarity
BLOCK_SIZE = 2048       # block size for similarity computation


# -----------------------------
# HELPERS
# -----------------------------
def slugify(s: str) -> str:
    """Make a safe column-name suffix from category labels."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


def compute_topk_scores(para_emb: np.ndarray, term_emb: np.ndarray, topk: int, write_max: bool):
    """
    Compute paragraph-level (topk mean) similarity and optionally max similarity
    against a given set of term embeddings.

    Assumes embeddings are L2-normalized; cosine similarity = dot product.
    """
    n_paras = para_emb.shape[0]
    n_terms = term_emb.shape[0]

    topk_mean = np.empty(n_paras, dtype=np.float32)
    topk_max = np.empty(n_paras, dtype=np.float32) if write_max else None

    term_T = term_emb.T  # (d, n_terms)

    k = min(topk, n_terms)
    for start in range(0, n_paras, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_paras)
        S = para_emb[start:end] @ term_T  # (block, n_terms)

        topk_vals = np.partition(S, -k, axis=1)[:, -k:]
        topk_mean[start:end] = topk_vals.mean(axis=1)

        if write_max:
            topk_max[start:end] = S.max(axis=1)

    return topk_mean, topk_max


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading data...")
    df = pd.read_csv(TEXT_CSV, low_memory=False)
    lex = pd.read_csv(LEXICON_CSV)

    for col, path in [(TEXT_COLUMN, TEXT_CSV), (LEXICON_COLUMN, LEXICON_CSV), (CATEGORY_COLUMN, LEXICON_CSV)]:
        if col not in (df.columns if path == TEXT_CSV else lex.columns):
            raise ValueError(f"Column '{col}' not found in {path}")

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    # Clean lexicon
    lex_clean = lex[[LEXICON_COLUMN, CATEGORY_COLUMN]].copy()
    lex_clean[LEXICON_COLUMN] = (
        lex_clean[LEXICON_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    lex_clean[CATEGORY_COLUMN] = lex_clean[CATEGORY_COLUMN].astype(str).str.strip()

    lex_clean = lex_clean.replace({"": np.nan})
    lex_clean = lex_clean.dropna(subset=[LEXICON_COLUMN, CATEGORY_COLUMN])

    # Unique terms overall
    all_terms = lex_clean[LEXICON_COLUMN].unique().tolist()

    # Group terms by category (unique per category)
    category_to_terms = (
        lex_clean.groupby(CATEGORY_COLUMN)[LEXICON_COLUMN]
        .apply(lambda s: sorted(set(s.tolist())))
        .to_dict()
    )

    print(f"Paragraphs: {len(texts)}")
    print(f"All lexicon terms: {len(all_terms)}")
    print(f"Categories: {len(category_to_terms)} -> {list(category_to_terms.keys())}")

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding ALL lexicon terms once...")
    # Embed all terms once and reuse by slicing indices for categories
    all_term_emb = model.encode(
        all_terms,
        batch_size=BATCH_SIZE_LEX,
        normalize_embeddings=True,
        show_progress_bar=True,
    )  # (n_all_terms, d)

    term_to_idx = {t: i for i, t in enumerate(all_terms)}

    print("Embedding paragraphs...")
    para_emb = model.encode(
        texts,
        batch_size=BATCH_SIZE_TEXT,
        normalize_embeddings=True,
        show_progress_bar=True,
    )  # (n_paras, d)

    # -----------------------------
    # 1) Overall score (all terms)
    # -----------------------------
    print("Computing OVERALL top-k cosine similarity (all terms)...")
    overall_mean, overall_max = compute_topk_scores(
        para_emb, all_term_emb, TOPK, WRITE_MAX
    )
    df[f"crisis_all_top{TOPK}"] = overall_mean
    if WRITE_MAX:
        df["crisis_all_max"] = overall_max

    # -----------------------------
    # 2) Per-category scores
    # -----------------------------
    print("Computing PER-CATEGORY top-k cosine similarity...")
    for cat, terms in category_to_terms.items():
        cat_slug = slugify(cat)
        idxs = [term_to_idx[t] for t in terms if t in term_to_idx]

        if not idxs:
            print(f"  - Skipping category '{cat}' (no valid terms after cleaning).")
            continue

        cat_emb = all_term_emb[idxs, :]
        cat_mean, cat_max = compute_topk_scores(
            para_emb, cat_emb, TOPK, WRITE_MAX
        )

        df[f"crisis_{cat_slug}_top{TOPK}"] = cat_mean
        if WRITE_MAX:
            df[f"crisis_{cat_slug}_max"] = cat_max

        print(f"  - Done: {cat} ({len(idxs)} terms) -> crisis_{cat_slug}_top{TOPK}" + (f", crisis_{cat_slug}_max" if WRITE_MAX else ""))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to: {OUTPUT_CSV}")
    print("Wrote columns:")
    cols = [c for c in df.columns if c.startswith("crisis_")]
    print("  " + ", ".join(cols))


if __name__ == "__main__":
    main()

