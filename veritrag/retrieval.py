import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """Lexical TF-IDF retriever. Similarity is lexical, not semantic."""

    def __init__(self, documents, min_similarity=0.01):
        if not documents:
            raise ValueError("No documents available")

        self.documents = documents
        self.min_similarity = float(min_similarity)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )
        texts = [doc["text"] for doc in documents]
        try:
            self.matrix = self.vectorizer.fit_transform(texts)
        except ValueError as error:
            raise ValueError(
                "The uploaded documents do not contain enough searchable text."
            ) from error

    def search(self, query, k=4, preferred_sources=None):
        """Return only passages with non-trivial lexical overlap.

        preferred_sources is retained for backward compatibility but deliberately
        does not boost a cited source: citations must not influence evidence ranking.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if k <= 0:
            return []

        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = np.argsort(-scores, kind="stable")
        results = []

        for index in indices:
            score = float(scores[index])
            if score < self.min_similarity:
                break
            results.append({
                **self.documents[int(index)],
                "similarity": score
            })
            if len(results) >= k:
                break

        return results
