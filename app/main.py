from pathlib import Path
from PIL import UnidentifiedImageError
from app.cnn_model import CIFAR10Classifier
from app.gan_model import MNISTDigitGenerator
from typing import Union
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()

cnn_classifier = CIFAR10Classifier(
    Path(__file__).resolve().parent
    / "models"
    / "cifar10_cnn.pth"
)

gan_generator = MNISTDigitGenerator(
    Path(__file__).resolve().parent
    / "models"
    / "mnist_gan_generator.pth"
)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str = Field(min_length=1, examples=["apple"])

class SimilarityRequest(BaseModel):
    word1: str = Field(min_length=1, examples=["apple"])
    word2: str = Field(min_length=1, examples=["orange"])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)
    if embedding is None:
        raise HTTPException(status_code=404, detail=f"No embedding found for '{request.word}'")
    return {"word": request.word, "dimension": len(embedding), "embedding": embedding}

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = embedding_model.calculate_similarity(request.word1, request.word2)
    if similarity is None:
        raise HTTPException(status_code=404, detail="At least one of the words has no embedding")
    return {"word1": request.word1, "word2": request.word2, "similarity": similarity}

@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    image_bytes = file.file.read()

    try:
        return cnn_classifier.predict(image_bytes)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

@app.get(
    "/generate_digits",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}}
)
def generate_digits(
    num_samples: int = Query(16, ge=1, le=64),
    seed: int | None = Query(None, ge=0)
):
    # The same seed always returns the same digits
    png_bytes = gan_generator.generate_png(num_samples, seed)
    return Response(content=png_bytes, media_type="image/png")
