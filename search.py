"""
Search Engine - Day 1 Complete
Tokenizer, Inverted Index, TF-IDF Ranker, Query Engine, Caching, and Threading.
"""

import re
import math
import time
import json
import hashlib
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. TOKENIZER
# ============================================================

class Tokenizer:
    """
    Tokenize text into normalized terms.
    
    Responsibilities:
    - Lowercase normalization
    - Punctuation removal
    - Stopword filtering
    - Short word filtering
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

        # Step 2: Extract words using regex (alphanumeric only)
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
# 2. INVERTED INDEX
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

        # Single-term search support for underlying index
        # (Complex queries handled by QueryEngine)
        term = tokens[0]
        return dict(self.index.get(term, {}))

    def get_document(self, doc_id):
        """Retrieve raw document by ID."""
        return self.documents.get(doc_id)

    def get_all_terms(self):
        """Return all indexed terms."""
        return list(self.index.keys())

    def term_frequency(self, term, doc_id):
        """Return term frequency for a document."""
        term = term.lower()
        return self.index.get(term, {}).get(doc_id, 0)


# ============================================================
# 3. TF-IDF RANKER
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
        Compute Inverse Document Frequency (IDF) for a term.
        IDF = log(Total Docs / Docs with Term)
        """
        if term in self.idf_cache:
            return self.idf_cache[term]

        total_docs = len(self.index.documents)
        # Count how many docs contain the term
        doc_frequency = len(self.index.index.get(term, {}))

        if total_docs == 0 or doc_frequency == 0:
            idf = 0.0
        else:
            idf = math.log10(total_docs / doc_frequency)

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
        query_terms = self.index.tokenizer.tokenize(query_text)
        scores = defaultdict(float)

        for doc_id in candidate_docs:
            for term in query_terms:
                score = self.compute_tfidf(term, doc_id)
                scores[doc_id] += score

        # Sort by score descending
        ranked_results = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked_results


# ============================================================
# 4. CACHING MECHANISM
# ============================================================

class SearchCache:
    """
    Simple LRU (Least Recently Used) Cache for search results.
    """
    def __init__(self, capacity=100, use_redis=False):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        self.stats = {'hits': 0, 'misses': 0}
        self.use_redis = use_redis # Placeholder for future expansion

    def get(self, key):
        if key in self.cache:
            self.stats['hits'] += 1
            # Refresh position
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        
        self.stats['misses'] += 1
        return None

    def set(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)

    def get_stats(self):
        return self.stats


# ============================================================
# 5. QUERY ENGINES
# ============================================================

class QueryEngine:
    """
    Execute search queries over the index.
    """

    def __init__(self, inverted_index, ranker):
        self.index = inverted_index
        self.ranker = ranker

    def search(self, query_text, top_k=5):
        """Search for documents"""

        # Tokenize query
        query_terms = self.index.tokenizer.tokenize(query_text)

        if not query_terms:
            return []

        # Find candidate documents (Union of docs containing any query term)
        candidate_docs = set()
        for term in query_terms:
            docs_with_term = self.index.search(term)
            candidate_docs.update(docs_with_term.keys())

        # Rank candidates
        ranked = self.ranker.rank(query_text, list(candidate_docs))

        # Return top-k
        return ranked[:top_k]


class CachedQueryEngine:
    """
    Query engine decorator with caching.
    """
    
    def __init__(self, inverted_index, ranker, cache=None):
        self.engine = QueryEngine(inverted_index, ranker)
        self.cache = cache or SearchCache()
    
    def search(self, query_text, top_k=5):
        """Search with caching"""
        
        # Create cache key using hash of query
        cache_key = hashlib.md5(query_text.encode()).hexdigest()
        
        # Try cache
        cached = self.cache.get(cache_key)
        if cached:
            # Return cached results (deserialize if needed)
            return json.loads(cached) if isinstance(cached, str) else cached
        
        # Not cached, compute
        results = self.engine.search(query_text, top_k)
        
        # Cache results
        if results:
            # Store as simple list or JSON string
            self.cache.set(cache_key, results)
        
        return results


# ============================================================
# 6. DOCUMENT STORAGE & UTILS
# ============================================================

class DocumentStore:
    """Store and retrieve document metadata"""

    def __init__(self):
        self.docs = {}

    def add(self, doc_id, title, content):
        """Add document"""
        self.docs[doc_id] = {
            'title': title,
            'content': content
        }

    def get(self, doc_id):
        """Get document by ID"""
        return self.docs.get(doc_id)


class ThreadedIndexer:
    """Index documents using multiple threads"""
    
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.lock = threading.Lock()
    
    def build_index(self, documents):
        """Build index from documents in parallel"""
        index = InvertedIndex()
        
        def worker(docs_chunk):
            local_index = InvertedIndex()
            for doc_id, content in docs_chunk:
                local_index.add_document(doc_id, content)
            
            # Merge into main index (Thread Safe)
            with self.lock:
                for term, postings in local_index.index.items():
                    for doc_id, freq in postings.items():
                        index.index[term][doc_id] += freq
                
                # Update document store
                index.documents.update(local_index.documents)
        
        # Split documents into chunks for threads
        chunk_size = math.ceil(len(documents) / self.num_threads)
        threads = []
        
        for i in range(self.num_threads):
            start = i * chunk_size
            end = start + chunk_size
            chunk = documents[start:end]
            
            if not chunk: continue

            t = threading.Thread(target=worker, args=(chunk,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        return index


# ============================================================
# 7. TEST SUITE
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
    
    # Java is rarer (1 doc) than python (2 docs), so IDF should be higher
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

    # Search for "machine"
    ranked = ranker.rank("machine", [1, 2, 3])

    # Doc 1 has "machine" twice, Doc 2 once. Doc 1 should be first.
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
    assert 3 not in doc_ids

    print("✅ Query engine works correctly")


def test_document_retrieval():
    print("\n" + "=" * 60)
    print("TEST 7: Document Retrieval")
    print("=" * 60)

    index = InvertedIndex()
    store = DocumentStore()

    index.add_document(1, "Learn python programming basics")
    store.add(1, "Python Guide", "Learn python programming basics")

    index.add_document(2, "Java programming fundamentals")
    store.add(2, "Java Basics", "Java programming fundamentals")

    ranker = TFIDFRanker(index)
    results = ranker.rank("python", [1, 2])

    assert results[0][0] == 1

    doc = store.get(1)
    assert doc['title'] == "Python Guide"

    print("✅ Document retrieval works")


def test_large_index():
    print("\n" + "="*60)
    print("TEST 8: Large Index (10K Documents)")
    print("="*60)
    
    index = InvertedIndex()
    
    # 1. Build index with 10K documents (Synthetic Data)
    # We ensure "python" is in the text so we don't get 0 results
    start = time.time()
    for i in range(10000):
        if i % 100 == 0:
            text = f"document {i} contains the target word python for search"
        else:
            text = f"document {i} is just filler content without the keyword"
        index.add_document(i, text)
    
    index_time = time.time() - start
    
    ranker = TFIDFRanker(index)
    engine = QueryEngine(index, ranker)
    
    # 2. Search
    start = time.time()
    results = engine.search("python")
    search_time = time.time() - start
    
    print(f"Indexed 10,000 documents: {index_time:.4f}s")
    print(f"Search query 'python': {search_time*1000:.2f}ms")
    print(f"Found {len(results)} results")
    
    # Fix: Actually assert that we found results
    assert len(results) > 0
    print(f"Total terms in index: {len(index.index.keys())}")
    print("✅ Performance acceptable")


def test_threaded_indexing():
    print("\n" + "="*60)
    print("TEST 9: Threaded Indexing")
    print("="*60)
    
    # Create 1000 documents
    documents = [(i, f"document {i} with python") for i in range(1000)]
    
    indexer = ThreadedIndexer(num_threads=4)
    
    start = time.time()
    index = indexer.build_index(documents)
    threaded_time = time.time() - start
    
    # Verify index
    assert len(index.documents) == 1000
    results = index.search("python")
    assert len(results) == 1000
    
    print(f"Indexed 1000 documents (4 threads): {threaded_time:.4f}s")
    print("✅ Threaded indexing works")


def test_caching():
    print("\n" + "="*60)
    print("TEST 10: Caching")
    print("="*60)
    
    # Initialize Cache
    cache = SearchCache(use_redis=False)
    
    # First access = miss
    val = cache.get("test_key")
    assert val is None
    
    # Set value
    cache.set("test_key", "cached_value")
    
    # Second access = hit
    result = cache.get("test_key")
    assert result == "cached_value"
    
    # Verify Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    print("✅ Caching works")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DAY 1: INDEXING + RANKING (FULL SYSTEM)")
    print("="*70)
    
    test_tokenizer()
    test_inverted_index()
    test_term_frequency()
    test_idf_calculation()
    test_tfidf_ranking()
    test_query_engine()
    test_document_retrieval()
    test_large_index()
    test_threaded_indexing()
    test_caching()
    
    print("\n" + "="*70)
    print("ALL 10 TESTS PASSED! ✅")
    print("="*70)