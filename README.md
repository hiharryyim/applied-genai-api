# applied-genai-api

FastAPI app for Applied Generative AI.

- `/generate`: bigram text generator (Module 3 class activity)
- `/embedding`, `/similarity`: word embeddings with spaCy `en_core_web_lg` (Assignment 1)

## Run with Docker

```bash
docker build -t gentext-app .
docker run -p 8000:80 gentext-app
```

The first build downloads the spaCy model (~400 MB), so it takes a few minutes.

Then open http://127.0.0.1:8000/docs

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run fastapi dev app/main.py
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | `{"Hello": "World"}` |
| POST | `/generate` | `{"start_word": "the", "length": 12}` | generated text |
| POST | `/embedding` | `{"word": "apple"}` | 300-dim vector for the word |
| POST | `/similarity` | `{"word1": "apple", "word2": "orange"}` | cosine similarity |

`/embedding` and `/similarity` return 404 if a word is not in the model's vocabulary.

Example:

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```
