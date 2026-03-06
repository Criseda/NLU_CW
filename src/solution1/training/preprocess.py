import re
import unicodedata
import pickle
import argparse
from pathlib import Path

import pandas as pd
import spacy
from joblib import Parallel, delayed
from tqdm import tqdm



def preprocess_text(text: str) -> str:
    """
    Minimal preprocessing: strip whitespace + NFKC normalisation only.
    We do NOT lowercase, remove punctuation, or remove stopwords as these are primary stylometric signals.
    """
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text

_nlp = None  # module-level cache

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


def process_pair(row_dict: dict) -> dict:
    """
    Run spaCy on text_1 and text_2 for a single pair.
    Returns a dict with cleaned texts, sentences, tokens, and POS tags.
    Designed to be called in parallel via joblib.
    """

    nlp = _get_nlp()

    result = {"id": row_dict.get("id", None), "label": row_dict.get("label", None)}

    for key in ("text_1", "text_2"):
        raw = preprocess_text(row_dict[key])
        doc = nlp(raw)

        sentences = [sent.text for sent in doc.sents]
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        words = [token.text for token in doc if not token.is_punct and not token.is_space]

        result[key] = {
            "raw": raw,
            "sentences": sentences,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "words": words,
        }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV (e.g. train.csv)")
    parser.add_argument("--output", required=True, help="Path to output .pkl cache file")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers (use 32 on CSF)")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only process first N rows (for testing)")
    args = parser.parse_args()

    print(f"Loading data from {args.input} ...")
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    print(f"Loading spaCy model ...")
    # We only need tokeniser, sentenciser, POS tagger

    rows = df.to_dict(orient="records")

    print(f"Processing {len(rows)} pairs with {args.n_jobs} workers ...")
    processed = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_pair)(row) for row in tqdm(rows)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(processed)} processed pairs to {output_path}")


if __name__ == "__main__":
    main()
