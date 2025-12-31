"""
Search Engine - Day 1 Commit 3
Tokenizer, Inverted Index, TF-IDF Ranker, and Query Engine
"""

import re
import math
from collections import defaultdict


# ============================================================
# TOKENIZER
# ============================================================

class Tokenizer:
    """
    Tokenize text into normalized terms.

    Responsibilities:
    - lowercase normalization
    - punctuation removal
    - stopword filtering
    """

    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'it', 'this', 'that', 'these', 'those',
        'over'
    }

    def tokenize(self, text):
        """
        Convert raw text into a list of normalized tokens.
        """

        # Step 1: Normalize case
        text = text.lower()

        # Step 2: Extract words using regex
        words = re.findall(r'\b\w+\b', text)

        # Step 3: Filter stopwords and short tokens
        terms = []

        for word in words:
            if word in self.STOPWORDS:
                continue

            if len(word) <= 1:
                continue

            terms.append(word)

        return terms


# ============================================================
# INVERTED INDEX
# ============================================================

class InvertedIndex:
    """
    Build and query an inverted index.

    Data structures:
    - index: term -> {doc_id -> term frequency}
    - documents: doc_id -> raw document content
    """

    def __init__(self):
        # term -> {doc_id -> frequency}
        self.index = defaultdict(lambda: defaultdict(int))

        # doc_id -> content
        self.documents = {}

        # shared tokenizer
        self.tokenizer = Tokenizer()

    def add_document(self, doc_id, content):
        """
        Add a document to the index.
        """

        # Store document content
        self.documents[doc_id] = content

        # Tokenize content
        terms = self.tokenizer.tokenize(content)

        # Update inverted index
        for term in terms:
            self.index[term][doc_id] += 1

    def search(self, query):
        """
        Return documents containing the query term.
        """

        tokens = self.tokenizer.tokenize(query)

        if not tokens:
            return {}

        # Single-term search (Day 1 scope)
        term = tokens[0]

        return dict(self.index.get(term, {}))

    def get_document(self, doc_id):
        """
        Retrieve raw document by ID.
        """
        return self.documents.get(doc_id)

    def get_all_terms(self):
        """
        Return all indexed terms.
        """
        return list(self.index.keys())

    def term_frequency(self, term, doc_id):
        """
        Return term frequency for a document.
        """
        term = term.lower()
        return self.index.get(term, {}).get(doc_id, 0)


# ============================================================
# TF-IDF RANKER
# ============================================================

class TFIDFRanker:
    """
    Rank documents using TF-IDF scoring.
    """

    def __init__(self, inverted_index):
        self.index = inverted_index
        self.idf_cache = {}

    def compute_idf(self, term):
        """
        Compute inverse document frequency for a term.
        """

        if term in self.idf_cache:
            return self.idf_cache[term]

        total_docs = len(self.index.documents)
        doc_frequency = len(self.index.index.get(term, {}))

        if total_docs == 0 or doc_frequency == 0:
            idf = 0.0
        else:
            idf = math.log(total_docs / doc_frequency)

        self.idf_cache[term] = idf
        return idf

    def compute_tfidf(self, term, doc_id):
        """
        Compute TF-IDF score for a term in a document.
        """

        tf = self.index.term_frequency(term, doc_id)
        idf = self.compute_idf(term)

        return tf * idf

    def rank(self, query_text, candidate_docs):
        """
        Rank candidate documents for a query.
        """

        # Tokenize query inside ranker (single source of truth)
        query_terms = self.index.tokenizer.tokenize(query_text)

        scores = {}

        for doc_id in candidate_docs:
            total_score = 0.0

            for term in query_terms:
                total_score += self.compute_tfidf(term, doc_id)

            if total_score > 0:
                scores[doc_id] = total_score

        ranked_results = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked_results


# ============================================================
# QUERY ENGINE
# ============================================================

class QueryEngine:
    """
    Execute search queries over the index.
    """

    def __init__(self, inverted_index, ranker):
        self.index = inverted_index
        self.ranker = ranker

    def search(self, query_text, top_k=5):
        """
        Execute a ranked search query.
        """

        # Tokenize query
        query_terms = self.index.tokenizer.tokenize(query_text)

        if not query_terms:
            return []

        # Collect candidate documents (OR semantics)
        candidate_docs = set()

        for term in query_terms:
            docs = self.index.search(term)
            candidate_docs.update(docs.keys())

        # Rank candidates
        ranked = self.ranker.rank(query_text, list(candidate_docs))

        # Return top-k results
        return ranked[:top_k]


# ============================================================
# TESTS
# ============================================================

def test_tokenizer():
    print("\n" + "=" * 60)
    print("TEST 1: Tokenizer")
    print("=" * 60)

    tokenizer = Tokenizer()
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.tokenize(text)

    expected = ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
    assert tokens == expected

    print("✅ Tokenizer works correctly")


def test_inverted_index():
    print("\n" + "=" * 60)
    print("TEST 2: Inverted Index")
    print("=" * 60)

    index = InvertedIndex()

    index.add_document(1, "python is great")
    index.add_document(2, "python java comparison")
    index.add_document(3, "java development tools")

    results = index.search("python")

    assert 1 in results
    assert 2 in results
    assert 3 not in results

    print("✅ Inverted index works correctly")


def test_term_frequency():
    print("\n" + "=" * 60)
    print("TEST 3: Term Frequency")
    print("=" * 60)

    index = InvertedIndex()

    index.add_document(1, "machine learning machine learning deep learning")
    index.add_document(2, "machine learning basics")

    assert index.term_frequency("machine", 1) == 2
    assert index.term_frequency("machine", 2) == 1

    print("✅ Term frequency calculation correct")


def test_idf_calculation():
    print("\n" + "=" * 60)
    print("TEST 4: IDF Calculation")
    print("=" * 60)

    index = InvertedIndex()

    index.add_document(1, "python programming")
    index.add_document(2, "python tutorial")
    index.add_document(3, "java development")

    ranker = TFIDFRanker(index)

    idf_python = ranker.compute_idf("python")
    idf_java = ranker.compute_idf("java")

    assert idf_java > idf_python

    print("✅ IDF calculation correct")


def test_tfidf_ranking():
    print("\n" + "=" * 60)
    print("TEST 5: TF-IDF Ranking")
    print("=" * 60)

    index = InvertedIndex()

    index.add_document(1, "machine learning machine")
    index.add_document(2, "machine learning basics")
    index.add_document(3, "deep learning only")

    ranker = TFIDFRanker(index)

    ranked = ranker.rank("machine", [1, 2, 3])

    assert ranked[0][0] == 1
    assert ranked[1][0] == 2

    print("✅ TF-IDF ranking correct")


def test_query_engine():
    print("\n" + "=" * 60)
    print("TEST 6: Query Engine")
    print("=" * 60)

    index = InvertedIndex()

    index.add_document(1, "python programming language")
    index.add_document(2, "java programming basics")
    index.add_document(3, "web development tutorials")

    ranker = TFIDFRanker(index)
    engine = QueryEngine(index, ranker)

    results = engine.search("programming")

    doc_ids = [doc_id for doc_id, _ in results]

    assert 1 in doc_ids
    assert 2 in doc_ids

    print("✅ Query engine works correctly")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DAY 1 COMMIT 3: QUERY ENGINE")
    print("=" * 70)

    test_tokenizer()
    test_inverted_index()
    test_term_frequency()
    test_idf_calculation()
    test_tfidf_ranking()
    test_query_engine()

    print("\n" + "=" * 70)
    print("ALL 6 TESTS PASSED! ✅")
    print("=" * 70)
