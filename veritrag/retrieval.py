
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class Retriever:

    def __init__(self, documents):

        if not documents:
            raise ValueError("No documents available")

        self.documents = documents

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )

        texts = [
            doc["text"]
            for doc in documents
        ]

        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=4, preferred_sources=None):

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).ravel()

        indices = np.argsort(scores)[::-1]

        results = []

        for index in indices:

            document = self.documents[int(index)]

            results.append({

                **document,

                "similarity": float(scores[index])

            })

        preferred_sources = set(preferred_sources or [])

        if preferred_sources:

            results.sort(
                key=lambda x: (
                    x["source"] in preferred_sources,
                    x["similarity"]
                ),
                reverse=True
            )

        return results[:k]
