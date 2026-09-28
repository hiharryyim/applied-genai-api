import random
import re
from collections import Counter, defaultdict


class BigramModel:
    """Bigram text generator, adapted from the Module 2 Word Sampling practical."""

    def __init__(self, corpus):
        self.vocab, self.bigram_probs = self.analyze_bigrams(corpus)

    @staticmethod
    def simple_tokenizer(text):
        """Lowercase the text and split it into word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def analyze_bigrams(self, corpus):
        """Compute p(w2 | w1) = C(w1, w2) / C(w1) over every document in the corpus."""
        bigram_counts = Counter()
        unigram_counts = Counter()
        for text in corpus:
            words = self.simple_tokenizer(text)
            # Count bigrams per document so they never span two unrelated sentences
            bigram_counts.update(zip(words[:-1], words[1:]))
            unigram_counts.update(words)

        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]
        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=10):
        """Generate up to `length` words by sampling each next word from the bigram probabilities."""
        current_word = start_word.lower()
        generated_words = [current_word]
        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # No known continuation, so stop early
                break
            current_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(current_word)
        return " ".join(generated_words)