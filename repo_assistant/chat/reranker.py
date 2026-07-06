"""
reranker.py

Implements two-stage retrieval using a Cross-Encoder for reranking.
"""

from typing import List
from sentence_transformers import CrossEncoder

from repo_assistant.store.models import SearchResult


class CrossEncoderReranker:
    """
    Reranks a list of initial search results by passing the query and the 
    document context together through a Cross-Encoder model. This provides
    significantly higher accuracy than Bi-Encoder cosine similarity alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        # Loads the cross-encoder model (downloads on first use)
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """
        Takes the initial results, scores them using the Cross-Encoder,
        and returns the top_k best matches.

        Args:
            query: The user's original question.
            results: The list of candidate SearchResults from the VectorStore.
            top_k: The number of results to keep after reranking.

        Returns:
            A pruned and sorted list of the top_k SearchResults.
        """
        if not results:
            return []

        # The CrossEncoder expects a list of [query, document] pairs
        pairs = [[query, r.chunk.content] for r in results]
        
        # Predict returns an array of scores corresponding to each pair
        scores = self.model.predict(pairs)

        # Update the scores and sort descending
        for res, score in zip(results, scores):
            res.score = float(score)

        # Re-sort based on the new Cross-Encoder scores
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]
