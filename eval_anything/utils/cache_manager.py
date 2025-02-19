"""
Cache management utilities for storing and retrieving inference results.

Engine 解耦：
- 不同的 engine 产生的 raw output 有不同的数据结构
- 因此需要为每个 engine 提供一个专门的 serializer

MultiModality 
- 不同的 modality 有不同的 serializer

- 在 Serializer 和 Deserializer 中, 灵活组合不同engine和modality
- CacheManager 只负责路径生成、存储和读取的逻辑
"""

import os
import json
import hashlib
from typing import List, Tuple

from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.serializers import SerializerRegistry

class CacheManager:
    """Cache manager for inference results"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
    def get_cache_path(self, model_cfg: dict, inputs: List[InferenceInput]) -> Tuple[str, bool]:
        """Get cache path and check if exists
        
        Args:
            model_cfg: Model configuration dictionary
            inputs: List of inference inputs
            
        Returns:
            Tuple of (cache_path, exists_flag)
        """
        model_name = model_cfg.get('model_name', 'unknown_model')
        
        # Create deterministic string representation of inputs
        input_strings = []
        for data in inputs:
            # Include task and text
            input_str = f"task:{data.task}|text:{data.text}"
            
            # Include UUID if present
            if data.uuid:
                uuid_str = '|'.join(f"{k}:{v}" for k, v in sorted(data.uuid.items()))
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
        cache_path = os.path.join(self.cache_dir, model_name, f"{inputs_hash}.json")
        
        return cache_path, os.path.exists(cache_path)

    def save(self, cache_path: str, outputs: List[InferenceOutput]) -> None:
        """Save outputs to cache"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cache_data = []
        for output in outputs:
            engine_serializer = SerializerRegistry.get(output.engine)
            cache_entry = engine_serializer.serialize(output)
            cache_data.append(cache_entry)
            
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def load(self, cache_path: str) -> List[InferenceOutput]:
        """Load outputs from cache"""
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        outputs = []
        for data in cache_data:
            serializer = SerializerRegistry.get(data['engine'])
            output = serializer.deserialize(data)
            outputs.append(output)
            
        return outputs