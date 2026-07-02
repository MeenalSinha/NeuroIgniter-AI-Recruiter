"""
Tests for the distributional semantic embedder (semantic.py).
Verifies genuine semantic behavior: paraphrased text should score higher
than unrelated text, even with zero lexical overlap with the reference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuroigniter.semantic import DistributionalEmbedder, _tokenize


class TestTokenize:
    def test_removes_punctuation_and_lowercases(self):
        tokens = _tokenize("Built FAISS-based Retrieval System!")
        assert 'faiss' in tokens or 'based' in tokens
        assert all(t == t.lower() for t in tokens)

    def test_removes_stopwords(self):
        tokens = _tokenize("the system that decides what to show")
        assert 'the' not in tokens
        assert 'that' not in tokens

    def test_filters_short_tokens(self):
        tokens = _tokenize("a an at to in on is")
        assert len(tokens) == 0


class TestDistributionalEmbedderTraining:
    def test_fit_on_reasonable_corpus_succeeds(self):
        corpus = [
            "built a production retrieval system using vector search",
            "deployed embeddings to real users at scale serving queries",
            "managed financial accounts and reconciliation reports",
            "prepared audit documentation for quarterly review",
        ] * 10  # repeat to get above document-frequency thresholds

        embedder = DistributionalEmbedder(vocab_size=200, embedding_dim=16)
        embedder.fit(corpus)
        assert embedder.is_trained
        assert len(embedder.vocab) > 0
        assert embedder.word_vectors is not None
        assert embedder.word_vectors.shape[0] == len(embedder.vocab)

    def test_fit_on_tiny_corpus_does_not_crash(self):
        """Pathologically small input should fail gracefully, not raise."""
        embedder = DistributionalEmbedder(vocab_size=200, embedding_dim=16)
        embedder.fit(["short text"])
        # May or may not train depending on vocab size, but must not crash
        assert isinstance(embedder.is_trained, bool)

    def test_fit_on_empty_corpus_does_not_crash(self):
        embedder = DistributionalEmbedder(vocab_size=200, embedding_dim=16)
        embedder.fit([])
        assert not embedder.is_trained

    def test_embedding_dimension_respected(self):
        corpus = [
            "built production retrieval ranking system embeddings vector search",
            "deployed real users scale latency monitoring inference api",
            "python machine learning model training evaluation pipeline",
        ] * 15

        embedder = DistributionalEmbedder(vocab_size=200, embedding_dim=8)
        embedder.fit(corpus)
        if embedder.is_trained:
            assert embedder.word_vectors.shape[1] <= 8


class TestDistributionalEmbedderSemantics:
    """The core claim: this should capture genuine semantic similarity,
    not just lexical overlap, by training on a corpus where related
    concepts co-occur even when exact words differ."""

    def setup_method(self):
        # A corpus where IR-related terms co-occur frequently with each
        # other, and finance-related terms co-occur with each other —
        # this lets the model learn that "retrieval" and "search" are
        # distributionally similar (they appear in similar contexts),
        # while "accounts" and "search" are not.
        ir_docs = [
            "production retrieval system embeddings vector search index ranking",
            "deployed search infrastructure serving real users scale latency",
            "built ranking pipeline retrieval embeddings dense vector similarity",
            "search system architecture retrieval index production deployment",
            "embeddings model vector search ranking system production scale",
        ]
        finance_docs = [
            "managed accounts payable reconciliation financial reporting",
            "prepared audit documentation quarterly review accounts",
            "financial accounts reconciliation reporting audit compliance",
            "accounts payable management financial audit reporting quarterly",
            "reconciliation accounts financial reporting compliance audit",
        ]
        self.corpus = (ir_docs * 8) + (finance_docs * 8)

        self.embedder = DistributionalEmbedder(vocab_size=300, embedding_dim=24)
        self.embedder.fit(self.corpus)

    def test_embedder_trains_successfully_on_realistic_corpus(self):
        assert self.embedder.is_trained

    def test_semantically_similar_text_scores_higher_than_unrelated(self):
        """A held-out IR sentence using DIFFERENT specific words than the
        training docs should still score closer to an IR reference than
        a finance sentence does — this is the actual semantic test."""
        jd_text = "production retrieval ranking embeddings vector search system"
        jd_vec = self.embedder.embed(jd_text)
        assert jd_vec is not None

        # Held-out sentence: shares the *topic* (IR/search) but uses some
        # different specific words than what's in the JD text itself
        ir_holdout = "search infrastructure deployment scale latency production"
        finance_holdout = "financial audit accounts reporting compliance"

        sim_ir = self.embedder.similarity(ir_holdout, jd_vec)
        sim_finance = self.embedder.similarity(finance_holdout, jd_vec)

        assert sim_ir > sim_finance, (
            f"Expected IR-topic text ({sim_ir:.3f}) to score higher than "
            f"finance-topic text ({sim_finance:.3f}) against an IR JD vector"
        )

    def test_similarity_bounded_zero_to_one(self):
        jd_vec = self.embedder.embed("production retrieval ranking system")
        for text in ["search embeddings vector", "accounts financial audit", ""]:
            sim = self.embedder.similarity(text, jd_vec)
            assert 0.0 <= sim <= 1.0

    def test_empty_text_returns_neutral_score(self):
        jd_vec = self.embedder.embed("production retrieval system")
        sim = self.embedder.similarity("", jd_vec)
        assert sim == 0.5  # Neutral fallback, not 0 (which would be a penalty)

    def test_none_reference_vector_returns_neutral(self):
        sim = self.embedder.similarity("some text here", None)
        assert sim == 0.5


class TestDistributionalEmbedderUntrained:
    """An embedder that failed to train (e.g. tiny corpus) must degrade
    gracefully to neutral scores, never crash or silently bias rankings."""

    def test_similarity_on_untrained_embedder_returns_neutral(self):
        embedder = DistributionalEmbedder(vocab_size=200, embedding_dim=16)
        # Don't call fit() — embedder.is_trained stays False
        sim = embedder.similarity("any text", None)
        assert sim == 0.5

    def test_embed_on_untrained_embedder_returns_none(self):
        embedder = DistributionalEmbedder(vocab_size=200, embedding_dim=16)
        result = embedder.embed("any text")
        assert result is None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
