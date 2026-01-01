"""
Interactive Search Engine Demo
"""

from search import InvertedIndex, TFIDFRanker, QueryEngine, CachedQueryEngine
from cache import SearchCache
import time


def demo_simple_search():
    """Demo 1: Simple Search"""
    print("\n" + "="*70)
    print("DEMO 1: Simple Search Query")
    print("="*70)
    
    index = InvertedIndex()
    index.add_document(1, "Python Programming Guide")
    index.add_document(2, "Python vs JavaScript")
    index.add_document(3, "Java Development Tools")
    
    ranker = TFIDFRanker(index)
    engine = QueryEngine(index, ranker)
    
    query = "python"
    results = engine.search(query)
    
    print(f"\nQuery: '{query}'")
    print(f"Results:")
    for doc_id, score in results:
        print(f"  Doc {doc_id}: {score:.2f}")


def demo_multiterm_search():
    """Demo 2: Multi-term Search"""
    print("\n" + "="*70)
    print("DEMO 2: Multi-term Search")
    print("="*70)
    
    index = InvertedIndex()
    index.add_document(1, "machine learning basics tutorial")
    index.add_document(2, "machine learning deep learning")
    index.add_document(3, "deep learning only")
    
    ranker = TFIDFRanker(index)
    engine = QueryEngine(index, ranker)
    
    query = "machine learning"
    results = engine.search(query)
    
    print(f"\nQuery: '{query}'")
    print(f"Results:")
    for doc_id, score in results:
        print(f"  Doc {doc_id}: {score:.2f}")


def demo_caching_impact():
    """Demo 3: Cache Impact"""
    print("\n" + "="*70)
    print("DEMO 3: Caching Impact")
    print("="*70)
    
    # Build larger index
    index = InvertedIndex()
    for i in range(10000):
        index.add_document(i, f"document {i} with python programming")
    
    ranker = TFIDFRanker(index)
    cache = SearchCache(use_redis=False)
    engine = CachedQueryEngine(index, ranker, cache)
    
    query = "python"
    
    # First query (cache miss)
    start = time.time()
    results = engine.search(query)
    first_time = (time.time() - start) * 1000
    
    # Second query (cache hit)
    start = time.time()
    results = engine.search(query)
    second_time = (time.time() - start) * 1000
    
    print(f"\nQuery: '{query}'")
    print(f"First call (cache miss): {first_time:.1f}ms")
    print(f"Second call (cache hit): {second_time:.2f}ms")
    print(f"Speedup: {first_time / second_time:.0f}x")


def demo_concurrent_queries():
    """Demo 4: Concurrent Queries"""
    print("\n" + "="*70)
    print("DEMO 4: Concurrent Query Simulation")
    print("="*70)
    
    import random
    
    index = InvertedIndex()
    for i in range(50000):
        index.add_document(i, f"document {i} with python")
    
    ranker = TFIDFRanker(index)
    cache = SearchCache(use_redis=False)
    engine = CachedQueryEngine(index, ranker, cache)
    
    queries = ["python", "programming", "tutorial"]
    
    start = time.time()
    for _ in range(1000):
        q = random.choice(queries)
        engine.search(q)
    
    elapsed = time.time() - start
    throughput = 1000 / elapsed
    
    print(f"\n1000 queries completed in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.0f} queries/sec")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SEARCH ENGINE INTERACTIVE DEMO")
    print("="*70)
    
    demo_simple_search()
    demo_multiterm_search()
    demo_caching_impact()
    demo_concurrent_queries()
    
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED! ✅")
    print("="*70)
