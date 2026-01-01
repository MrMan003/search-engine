"""
Redis Caching Layer
"""

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class SearchCache:
    """Cache search results in Redis"""
    
    def __init__(self, use_redis=False, host='localhost', port=6379):
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.hits = 0
        self.misses = 0
        
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
                self.redis_client.ping()
            except:
                self.use_redis = False
    
    def get(self, key):
        """Get from cache"""
        if not self.use_redis:
            return None
        
        try:
            result = self.redis_client.get(key)
            if result:
                self.hits += 1
            else:
                self.misses += 1
            return result
        except:
            return None
    
    def set(self, key, value, ttl=3600):
        """Set in cache with TTL"""
        if not self.use_redis:
            return
        
        try:
            self.redis_client.setex(key, ttl, value)
        except:
            pass
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
