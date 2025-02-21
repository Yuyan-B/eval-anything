"""
Cache management utilities for storing and retrieving inference results.

Use abstract cache keys instead of concrete file paths for `save` and `load` interfaces,
so that the cache manager is independent of the file system. This is convenient for 
switching to other storage backends in the future.

Use pickle to automatically handle the serialization and deserialization of complex objects.
"""

import hashlib
from typing import List, Tuple, Optional, Any
import logging
import pickle
from pathlib import Path

from eval_anything.utils.data_type import InferenceInput, InferenceOutput

class BinaryCache:
    """Binary cache implementation with support for multiple tensor types"""
    
    def __init__(self, cache_dir: str = ".cache/eval_anything"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path from key"""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.pkl"

    def is_cached(self, key: str) -> bool:
        """Check if the key is cached"""
        cache_path = self._get_cache_path(key)
        return cache_path.exists()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve object from cache"""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            logging.debug(f"Cache miss for key: {key}")
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logging.debug(f"Cache hit for key: {key}")
            return data
        except Exception as e:
            logging.error(f"Failed to load cached object with key {key}. Error: {e}")
            return None
            
    def put(self, key: str, value: Any) -> bool:
        """Store object in cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logging.debug(f"Successfully cached object with key: {key}")
            return True
        except Exception as e:
            logging.error(f"Failed to cache object with key {key}. Error: {e}")
            return False

    def clear(self):
        """Clear all cache files"""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logging.error(f"Failed to delete cache file {cache_file}. Error: {e}")
        logging.debug(f"Cleared {count} cache files from {self.cache_dir}")

class CacheManager:
    """Cache manager for inference results"""
    
    def __init__(self, cache_dir: str):
        """Initialize cache manager
        
        Args:
            cache_dir: Directory path for caching results
        """
        self.cache_dir = cache_dir
        self.binary_cache = BinaryCache(cache_dir)
        
    def get_cache_path(self, model_cfg: dict, inputs: List[InferenceInput]) -> Tuple[str, bool]:
        """Get cache path and check if exists
        
        Args:
            model_cfg: Model configuration dictionary
            inputs: List of inference inputs
            
        Returns:
            Tuple of (cache_key, exists_flag)
        """
        model_name = model_cfg.get('model_name', 'unknown_model')
        
        # Create deterministic string representation of inputs
        input_strings = []
        for data in inputs:
            # Include task and text (if present)
            input_str = f"task:{data.task}"
            if data.text is not None:
                input_str += f"|text:{data.text}"
            
            # Include UUID if present
            if data.uuid:
                uuid_str = '|'.join(f"{k}:{self._normalize_value(v)}" 
                                  for k, v in sorted(data.uuid.items()))
                input_str += f"|uuid:{uuid_str}"
                
            # Include multimodal data if present
            if data.mm_data:
                urls = [item.url for item in data.mm_data]
                input_str += f"|urls:{','.join(urls)}"
                
            input_strings.append(input_str)
        
        # Sort for deterministic ordering
        input_strings.sort()
        
        # Create hash from concatenated string
        inputs_hash = hashlib.md5(''.join(input_strings).encode()).hexdigest()
        cache_key = f"{model_name}_{inputs_hash}"
        
        return cache_key, self.binary_cache.is_cached(cache_key)

    def _normalize_value(self, value: any) -> str:
        """Normalize values for consistent hashing"""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def save(self, cache_key: str, outputs: List[InferenceOutput]) -> None:
        """Save outputs to cache"""
        self.binary_cache.put(cache_key, outputs)

    def load(self, cache_key: str) -> List[InferenceOutput]:
        """Load outputs from cache"""
        return self.binary_cache.get(cache_key)

    def clear(self) -> None:
        """Clear all cached inference results"""
        self.binary_cache.clear()