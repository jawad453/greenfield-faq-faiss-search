# GreenField Farmers FAQ — FAISS Semantic Search

A small Python module that embeds each FAQ question with `sentence-transformers/all-MiniLM-L6-v2`, stores normalized vectors in a local FAISS index, and returns the three most similar FAQ records for a query.

## Files

- `faq.json` — FAQ records reused from Task 1.
- `build_index.py` — creates `faiss_index.index` from the questions in `faq.json`.
- `search_faq.py` — provides `search_faq(query)`.
- `test_search_faq.py` — checks that a known membership query retrieves the expected FAQ first.
- `requirements.txt` — Python dependencies.

## Setup

Use Python 3.10 or 3.11 for broad compatibility with the ML dependencies.

```bash
python -m venv .venv
```

Activate the environment, then install the packages:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS/Linux**
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build the index

Run this once after installing dependencies. The first run downloads the sentence-transformers model and may take a few minutes.

```bash
python build_index.py
```

This creates `faiss_index.index` in the project directory. Commit this generated file to the repository as required by the task. Rebuild it whenever `faq.json` changes.

## Search FAQs

```python
from search_faq import search_faq

matches = search_faq("How do I become a member of the farmers cooperative?")
for faq in matches:
    print(faq["question"], faq["similarity_score"])
    print(faq["answer"])
```

`search_faq(query)` returns up to three FAQ dictionaries ordered from most to least similar. Each result includes `question`, `answer`, and `similarity_score`.

## Run the test

Build the index first, then run:

```bash
pytest -v
```

The test verifies that a known membership query ranks the expected FAQ first and that three results are returned in descending similarity order.

## Notes

- The index uses FAISS inner-product search over normalized embeddings, which is equivalent to cosine similarity for these vectors.
- The binary index is generated locally and should be committed to GitHub; it is not included in this source ZIP because it must be generated with the specified model.
