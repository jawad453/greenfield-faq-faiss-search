"""Build a local FAISS index from the questions in faq.json."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "faq.json"
INDEX_PATH = BASE_DIR / "faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index() -> None:
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        faqs = json.load(file)

    if not isinstance(faqs, list) or not faqs:
        raise ValueError("faq.json must contain a non-empty list of FAQ objects.")
    if any(not isinstance(item, dict) or not item.get("question") or not item.get("answer") for item in faqs):
        raise ValueError("Every FAQ must be an object with non-empty 'question' and 'answer' fields.")

    model = SentenceTransformer(MODEL_NAME)
    questions = [item["question"] for item in faqs]
    vectors = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Built FAISS index with {index.ntotal} FAQs: {INDEX_PATH.name}")


if __name__ == "__main__":
    build_index()
