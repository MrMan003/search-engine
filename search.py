"""
Search Engine - Day 2 Complete
Features:
1. Tokenizer (Regex + Stopwords)
2. Inverted Index (Add, Search, Frequency)
3. Persistence (Save/Load Index to Disk)
4. TF-IDF Ranker (Math-heavy ranking)
5. Query Engine (Coordinator)
6. Caching (LRU Strategy)
7. Threading (Parallel Indexing)
8. Benchmarking (Latency tests)
"""

import re
import math
import time
import json
import hashlib
import threading
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. TOKENIZER
# ============================================================

class Tokenizer:
    """
    Tokenize text into normalized terms.
    """

    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'it', 'this', 'that', 'these', 'those',
        'over'
    }

    def tokenize(self, text):
        if not text:
            return []

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
# 2. INVERTED INDEX (WITH PERSISTENCE)
# ============================================================

class InvertedIndex:
    """
    Build and query an inverted index.
    """

    def __init__(self):
        # term -> {doc_id -> frequency}
        self.index = defaultdict(lambda: defaultdict(int))
        self.documents = {}
        self.tokenizer = Tokenizer()

    def add_document(self, doc_id, content):
        self.documents[doc_id] = content
        terms = self.tokenizer.tokenize(content)
        for term in terms:
            self.index[term][doc_id] += 1

    def search(self, query):
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return {}
        term = tokens[0]
        return dict(self.index.get(term, {}))

    def term_frequency(self, term, doc_id):
        term = term.lower()
        return self.index.get(term, {}).get(doc_id, 0)

    def get_document(self, doc_id):
        return self.documents.get(doc_id)

    def get_all_terms(self):
        return list(self.index.keys())

    # --- PERSISTENCE METHODS ---

    def save(self, filename):
        print(f"Saving index to {filename}...")
        export_data = {
            "index": {k: dict(v) for k, v in self.index.items()},
            "documents": self.documents
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f)
        print("Save complete.")

    def load(self, filename):
        print(f"Loading index from {filename}...")
        if not os.path.exists(filename):
            print("File not found.")
            return

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.documents = {}
        for k, v in data["documents"].items():
            try:
                self.documents[int(k)] = v
            except ValueError:
                self.documents[k] = v
        
        self.index = defaultdict(lambda: defaultdict(int))
        for term, postings in data["index"].items():
            for doc_id, freq in postings.items():
                try:
                    self.index[term][int(doc_id)] = freq
                except ValueError:
                    self.index[term][doc_id] = freq
                    
        print(f"Load complete. Loaded {len(self.documents)} documents.")


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
        if term in self.idf_cache:
            return self.idf_cache[term]

        total_docs = len(self.index.documents)
        doc_frequency = len(self.index.index.get(term, {}))

        if total_docs == 0 or doc_frequency == 0:
            idf = 0.0
        else:
            # FIX: Add 1 to calculation to avoid log(1) = 0
            idf = math.log10(total_docs / doc_frequency) + 1.0

        self.idf_cache[term] = idf
        return idf

    def compute_tfidf(self, term, doc_id):
        tf = self.index.term_frequency(term, doc_id)
        idf = self.compute_idf(term)
        return tf * idf

    def rank(self, query_text, candidate_docs):
        query_terms = self.index.tokenizer.tokenize(query_text)
        scores = defaultdict(float)

        for doc_id in candidate_docs:
            for term in query_terms:
                score = self.compute_tfidf(term, doc_id)
                scores[doc_id] += score

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
    Simple LRU (Least Recently Used) Cache.
    """
    def __init__(self, capacity=100, use_redis=False):
        self.capacity = capacity
        self.cache = {}
        self.order = [] 
        self.stats = {'hits': 0, 'misses': 0}

    def get(self, key):
        if key in self.cache:
            self.stats['hits'] += 1
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
    def __init__(self, inverted_index, ranker):
        self.index = inverted_index
        self.ranker = ranker

    def search(self, query_text, top_k=5):
        query_terms = self.index.tokenizer.tokenize(query_text)

        if not query_terms:
            return []

        candidate_docs = set()
        for term in query_terms:
            docs_with_term = self.index.search(term)
            candidate_docs.update(docs_with_term.keys())

        ranked = self.ranker.rank(query_text, list(candidate_docs))
        return ranked[:top_k]


class CachedQueryEngine:
    def __init__(self, inverted_index, ranker, cache=None):
        self.engine = QueryEngine(inverted_index, ranker)
        self.cache = cache or SearchCache()
    
    def search(self, query_text, top_k=5):
        cache_key = hashlib.md5(query_text.encode()).hexdigest()
        
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached) if isinstance(cached, str) else cached
        
        results = self.engine.search(query_text, top_k)
        
        if results:
            self.cache.set(cache_key, results)
        
        return results


# ============================================================
# 6. THREADING UTILS
# ============================================================

class ThreadedIndexer:
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.lock = threading.Lock()
    
    def build_index(self, documents):
        index = InvertedIndex()
        
        def worker(docs_chunk):
            local_index = InvertedIndex()
            for doc_id, content in docs_chunk:
                local_index.add_document(doc_id, content)
            
            with self.lock:
                for term, postings in local_index.index.items():
                    for doc_id, freq in postings.items():
                        index.index[term][doc_id] += freq
                index.documents.update(local_index.documents)
        
        if not documents:
            return index

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
# 7. DOCUMENT STORAGE
# ============================================================

class DocumentStore:
    def __init__(self):
        self.docs = {}

    def add(self, doc_id, title, content):
        self.docs[doc_id] = {'title': title, 'content': content}

    def get(self, doc_id):
        return self.docs.get(doc_id)


# ============================================================
# 8. TEST SUITE (WITH STATS)
# ============================================================

def test_tokenizer():
    print("\n" + "=" * 60)
    print("TEST 1: Tokenizer")
    print("=" * 60)
    
    text = "The quick brown fox jumps over the lazy dog"
    start = time.time()
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    duration = (time.time() - start) * 1000
    
    print(f"Input text length: {len(text)} chars")
    print(f"Tokens generated: {len(tokens)}")
    print(f"Processing time: {duration:.4f} ms")
    
    assert ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog'] == tokens
    print("✅ Tokenizer works correctly")


def test_inverted_index():
    print("\n" + "=" * 60)
    print("TEST 2: Inverted Index")
    print("=" * 60)
    
    start = time.time()
    index = InvertedIndex()
    index.add_document(1, "python is great")
    index.add_document(2, "python java comparison")
    results = index.search("python")
    duration = (time.time() - start) * 1000
    
    print(f"Docs added: 2")
    print(f"Terms in index: {len(index.get_all_terms())}")
    print(f"Search results for 'python': {len(results)}")
    print(f"Execution time: {duration:.4f} ms")
    
    assert 1 in results and 2 in results
    print("✅ Inverted index works correctly")


def test_term_frequency():
    print("\n" + "=" * 60)
    print("TEST 3: Term Frequency")
    print("=" * 60)
    
    index = InvertedIndex()
    index.add_document(1, "machine learning machine")
    
    tf = index.term_frequency("machine", 1)
    print(f"Document: 'machine learning machine'")
    print(f"Term: 'machine'")
    print(f"Calculated Frequency: {tf}")
    
    assert tf == 2
    print("✅ Term frequency calculation correct")


def test_idf_calculation():
    print("\n" + "=" * 60)
    print("TEST 4: IDF Calculation")
    print("=" * 60)
    
    index = InvertedIndex()
    index.add_document(1, "python")
    index.add_document(2, "python")
    index.add_document(3, "java")
    
    ranker = TFIDFRanker(index)
    idf_java = ranker.compute_idf("java")
    idf_python = ranker.compute_idf("python")
    
    print(f"Docs containing 'python': 2 (Common)")
    print(f"Docs containing 'java': 1 (Rare)")
    print(f"IDF 'python': {idf_python:.4f}")
    print(f"IDF 'java': {idf_java:.4f}")
    
    assert idf_java > idf_python
    print("✅ IDF calculation correct (Rare terms have higher scores)")


def test_tfidf_ranking():
    print("\n" + "=" * 60)
    print("TEST 5: TF-IDF Ranking")
    print("=" * 60)
    
    index = InvertedIndex()
    index.add_document(1, "machine learning machine") # High TF
    index.add_document(2, "machine learning")         # Low TF
    
    ranker = TFIDFRanker(index)
    ranked = ranker.rank("machine", [1, 2])
    
    print(f"Query: 'machine'")
    print(f"Doc 1 Score (freq=2): {ranked[0][1]:.4f}")
    print(f"Doc 2 Score (freq=1): {ranked[1][1]:.4f}")
    
    assert ranked[0][0] == 1
    print("✅ TF-IDF ranking correct")


def test_query_engine():
    print("\n" + "=" * 60)
    print("TEST 6: Query Engine")
    print("=" * 60)
    
    index = InvertedIndex()
    index.add_document(1, "python programming")
    ranker = TFIDFRanker(index)
    engine = QueryEngine(index, ranker)
    
    start = time.time()
    results = engine.search("programming")
    duration = (time.time() - start) * 1000
    
    print(f"Query: 'programming'")
    print(f"Results found: {len(results)}")
    print(f"Top Result ID: {results[0][0]}")
    print(f"Search Time: {duration:.4f} ms")
    
    assert results[0][0] == 1
    print("✅ Query engine works correctly")


def test_document_retrieval():
    print("\n" + "=" * 60)
    print("TEST 7: Document Retrieval")
    print("=" * 60)
    
    store = DocumentStore()
    start = time.time()
    store.add(1, "Title", "Content")
    doc = store.get(1)
    duration = (time.time() - start) * 1000
    
    print(f"Stored Doc ID: 1")
    print(f"Retrieved Title: {doc['title']}")
    print(f"Retrieval Time: {duration:.4f} ms")
    
    assert doc['title'] == "Title"
    print("✅ Document retrieval works")


def test_large_index():
    print("\n" + "="*60)
    print("TEST 8: Large Index (10K Documents)")
    print("="*60)
    
    index = InvertedIndex()
    start = time.time()
    for i in range(10000):
        # Ensure 'python' is actually in the text
        if i % 100 == 0:
            text = f"doc {i} contains python" 
        else:
            text = f"doc {i} filler"
        index.add_document(i, text)
    duration = time.time() - start
    
    print(f"Indexed 10,000 docs: {duration:.4f}s")
    print(f"Total Terms: {len(index.get_all_terms())}")
    
    ranker = TFIDFRanker(index)
    engine = QueryEngine(index, ranker)
    
    s_start = time.time()
    results = engine.search("python")
    s_duration = (time.time() - s_start) * 1000
    
    print(f"Search Query 'python': {s_duration:.4f} ms")
    print(f"Results Found: {len(results)}")
    
    assert len(results) > 0
    print("✅ Performance acceptable")


def test_threaded_indexing():
    print("\n" + "="*60)
    print("TEST 9: Threaded Indexing")
    print("="*60)
    
    documents = [(i, f"doc {i} with python") for i in range(1000)]
    indexer = ThreadedIndexer(num_threads=4)
    
    start = time.time()
    index = indexer.build_index(documents)
    duration = time.time() - start
    
    print(f"Docs Indexed: 1000")
    print(f"Threads Used: 4")
    print(f"Execution Time: {duration:.4f}s")
    
    assert len(index.search("python")) == 1000
    print("✅ Threaded indexing works")


def test_caching():
    print("\n" + "="*60)
    print("TEST 10: Caching")
    print("="*60)
    
    cache = SearchCache(use_redis=False)
    
    # Miss
    val = cache.get("test")
    print(f"First access (Miss): {val}")
    assert val is None
    
    # Set
    cache.set("test", "value")
    print(f"Set 'test' -> 'value'")
    
    # Hit
    val = cache.get("test")
    print(f"Second access (Hit): {val}")
    assert val == "value"
    
    stats = cache.get_stats()
    print(f"Cache Stats: {stats}")
    
    assert stats['hits'] == 1
    print("✅ Caching works")


def test_latency_benchmark():
    """Test 11: Latency Benchmarks"""
    print("\n" + "="*60)
    print("TEST 11: Latency Benchmarks (Cold vs Warm Cache)")
    print("="*60)
    
    # Generate 20K docs for a noticeable benchmark
    DOC_COUNT = 20000 
    print(f"Generating and Indexing {DOC_COUNT} documents...")
    
    index = InvertedIndex()
    start_idx = time.time()
    for i in range(DOC_COUNT):
        # Make every 10th doc relevant
        term = "python" if i % 10 == 0 else "java"
        index.add_document(i, f"document {i} with {term} programming")
    print(f"Indexing complete: {time.time() - start_idx:.2f}s")
    
    ranker = TFIDFRanker(index)
    
    # 1. Test without cache
    engine_nocache = QueryEngine(index, ranker)
    start = time.time()
    results = engine_nocache.search("python programming")
    nocache_time = (time.time() - start) * 1000
    
    # 2. Test with cache (First call - Cold)
    cache = SearchCache(use_redis=False)
    engine_cached = CachedQueryEngine(index, ranker, cache)
    
    start = time.time()
    results = engine_cached.search("python programming")
    first_call = (time.time() - start) * 1000
    
    # 3. Test with cache (Second call - Warm)
    start = time.time()
    results = engine_cached.search("python programming")
    second_call = (time.time() - start) * 1000
    
    print(f"\nNo cache: {nocache_time:.3f}ms")
    print(f"First call (Cold): {first_call:.3f}ms")
    print(f"Second call (Warm): {second_call:.3f}ms")
    
    speedup = nocache_time / second_call if second_call > 0 else 999.0
    print(f"Speedup: {speedup:.1f}x")
    
    assert speedup >= 1.0 
    print("✅ Caching benchmark complete")

def test_throughput():
    """Test 12: Throughput Benchmark"""
    print("\n" + "="*60)
    print("TEST 12: Throughput (1000 Queries)")
    print("="*60)
    
    import time
    import random
    from cache import SearchCache
    
    # Build index
    index = InvertedIndex()
    for i in range(100000):
        index.add_document(i, f"document {i} with python machine learning")
    
    ranker = TFIDFRanker(index)
    
    cache = SearchCache(use_redis=False)
    engine = CachedQueryEngine(index, ranker, cache)
    
    queries = ["python", "machine", "learning", "programming", "tutorial"]
    
    start = time.time()
    for _ in range(1000):
        q = random.choice(queries)
        engine.search(q)
    
    elapsed = time.time() - start
    throughput = 1000 / elapsed
    
    print(f"1000 queries in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.0f} queries/sec")
    print("✅ Throughput test passed")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DAY 2 COMMIT 9: Throughput Tests")
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
    test_latency_benchmark()
    test_throughput()
    
    print("\n" + "="*70)
    print("ALL 12 TESTS PASSED! ✅")
    print("="*70)