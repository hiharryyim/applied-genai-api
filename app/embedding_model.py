import spacy


class EmbeddingModel:
    """Word embeddings with spaCy, adapted from the Module 2 Word Embeddings practical."""

    def __init__(self, model_name="en_core_web_lg"):
        # Load once at startup: the large model takes a few seconds to load
        self.nlp = spacy.load(model_name)

    def calculate_embedding(self, word):
        """Return the embedding vector for `word`, or None if the word is out of vocabulary."""
        doc = self.nlp(word)
        if not doc.has_vector:
            return None
        return doc.vector.tolist()  # numpy array -> list so FastAPI can serialize it

    def calculate_similarity(self, word1, word2):
        """Return the cosine similarity between two words, or None if either has no vector."""
        doc1, doc2 = self.nlp(word1), self.nlp(word2)
        if not (doc1.has_vector and doc2.has_vector):
            return None
        return float(doc1.similarity(doc2))