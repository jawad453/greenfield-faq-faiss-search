"""Search GreenField Farmers FAQs with sentence-transformers and FAISS."""
import json
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "faq.json"
INDEX_PATH = BASE_DIR / "faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_resources():
    """Load the FAQ records, embedding model, and prebuilt FAISS index once."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run 'python build_index.py' first to create it."
        )
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        faqs = json.load(file)
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    if index.ntotal != len(faqs):
        raise ValueError("FAISS index and faq.json contain different numbers of entries. Rebuild the index.")
    return faqs, model, index


def search_faq(query: str) -> list[dict]:
    """Return up to three FAQ objects ranked by semantic similarity to query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    faqs, model, index = _load_resources()
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    vector = np.asarray(vector, dtype="float32")
    scores, positions = index.search(vector, min(3, len(faqs)))

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0:
            faq = dict(faqs[int(position)])
            faq["similarity_score"] = float(score)
            results.append(faq)
    return results
