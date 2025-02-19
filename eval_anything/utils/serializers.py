"""
Serializers for different engines and modalities.

TODO:
- Add serializers for other backends
- Add serializers for other modalities
"""

from functools import wraps
from typing import Dict, Type, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.outputs import RequestOutput, RequestMetrics, CompletionOutput, LoRARequest
from eval_anything.utils.data_type import InferenceOutput, MultiModalData, ModalityType


class ModalitySerializerRegistry:
    """Registry for modality-specific serializers"""
    _registry: Dict[str, 'BaseModalitySerializer'] = {}
    
    @classmethod
    def register(cls, modality: str):
        def decorator(serializer_cls):
            cls._registry[modality] = serializer_cls()
            return serializer_cls
        return decorator
    
    @classmethod
    def get(cls, modality: str) -> Optional['BaseModalitySerializer']:
        return cls._registry.get(modality, cls._registry.get('text'))

class BaseModalitySerializer(ABC):
    """Base class for modality-specific serializers"""
    @abstractmethod
    def serialize_input(self, mm_data: 'MultiModalData') -> dict:
        """Serialize modality-specific input data"""
        pass
        
    @abstractmethod
    def serialize_output(self, mm_data: 'MultiModalData') -> dict:
        """Serialize modality-specific output data"""
        pass
        
    @abstractmethod
    def deserialize_input(self, data: dict) -> 'MultiModalData':
        """Deserialize modality-specific input data"""
        pass
        
    @abstractmethod
    def deserialize_output(self, data: dict) -> 'MultiModalData':
        """Deserialize modality-specific output data"""
        pass

@ModalitySerializerRegistry.register('text')
class TextModalitySerializer(BaseModalitySerializer):
    def serialize_input(self, mm_data: 'MultiModalData') -> dict:
        return {
            'text': mm_data.text,
            'type': 'text'
        }
    
    def serialize_output(self, mm_data: 'MultiModalData') -> dict:
        return {
            'text': mm_data.text,
            'type': 'text'
        }
    
    def deserialize_input(self, data: dict) -> 'MultiModalData':
        return MultiModalData(url=None, file=data['text'], modality=ModalityType.TEXT) # TODO: Fix MultiModalData constructor

    def deserialize_output(self, data: dict) -> 'MultiModalData':
        return MultiModalData(url=None, file=data['text'], modality=ModalityType.TEXT) # TODO: Fix MultiModalData constructor

@ModalitySerializerRegistry.register('image')
class ImageModalitySerializer(BaseModalitySerializer):
    def serialize_input(self, mm_data: 'MultiModalData') -> dict:
        return {
            'url': mm_data.url,
            'type': 'image'
        }
    
    def serialize_output(self, mm_data: 'MultiModalData') -> dict:
        return {
            'url': mm_data.url,
            'type': 'image'
        }
    
    def deserialize_input(self, data: dict) -> 'MultiModalData':
        return MultiModalData(url=data['url'], file=None, modality=ModalityType.IMAGE) # TODO: Fix constructor, replace with actual file
        
    def deserialize_output(self, data: dict) -> 'MultiModalData':
        return MultiModalData(url=data['url'], file=None, modality=ModalityType.IMAGE) # TODO: Fix constructor, replace with actual file

class SerializerRegistry:
    """Registry for serializers"""
    _registry: Dict[str, 'BaseSerializer'] = {}
    
    @classmethod
    def register(cls, engine: str):
        """Decorator to register a serializer for a specific engine"""
        def decorator(serializer_cls):
            cls._registry[engine] = serializer_cls()
            return serializer_cls
        return decorator
    
    @classmethod
    def get(cls, engine: str) -> Optional['BaseSerializer']:
        """Get serializer for specified engine"""
        return cls._registry.get(engine, cls._registry.get('default'))

class BaseSerializer(ABC):
    """Base class for all serializers"""
   
    def serialize(self, output: 'InferenceOutput') -> dict:
        """Serialize InferenceOutput to a dictionary"""
        data = {
            'engine': output.engine,
            'task': output.task,
            'uuid': output.uuid,
            'response': output.response,
        }
        
        if hasattr(output, 'response_token_ids'):
            data['response_token_ids'] = output.response_token_ids

        # Add additional fields based on engine-specific data
        data.update(self._serialize_additional_fields(output))
        
        # TODO:modality specific data
        # An example implementation of modality specific data serialization
        if hasattr(output, 'mm_input_data') and output.mm_input_data:
            data['mm_input_data'] = []
            for mm_data in output.mm_input_data:
                modality = mm_data.get_modality()
                serializer = ModalitySerializerRegistry.get(modality)
                data['mm_input_data'].append(serializer.serialize_input(mm_data))
                
        if hasattr(output, 'mm_output_data') and output.mm_output_data:
            data['mm_output_data'] = []
            for mm_data in output.mm_output_data:
                modality = mm_data.get_modality()
                serializer = ModalitySerializerRegistry.get(modality)
                data['mm_output_data'].append(serializer.serialize_output(mm_data))

        return data

    def deserialize(self, data: dict) -> 'InferenceOutput':
        """Base deserialize method with common logic"""
        engine = self.get_engine_type()

        output = InferenceOutput(
            engine=engine,
            task=data['task'],
            uuid=data['uuid'],
            response=data['response'],
        )

        if 'response_token_ids' in data:
            setattr(output, 'response_token_ids', data['response_token_ids'])

        # TODO: modality specific data
        # An example implementation of modality specific data deserialization
        if 'mm_input_data' in data:
            output.mm_input_data = []
            for mm_data in data['mm_input_data']:
                modality = mm_data['modality']
                serializer = ModalitySerializerRegistry.get(modality)
                setattr(output, 'mm_input_data', serializer.deserialize(mm_data))
        if 'mm_output_data' in data:
            output.mm_output_data = []
            for mm_data in data['mm_output_data']:
                modality = mm_data['modality']
                serializer = ModalitySerializerRegistry.get(modality)
                setattr(output, 'mm_output_data', serializer.deserialize(mm_data))

        # Add additional fields based on engine-specific data
        output = self._deserialize_additional_fields(data, output)
        return output

    def get_engine_type(self) -> str:
        """Get the engine type for this serializer"""
        return self.__class__.__name__.replace('Serializer', '').lower()

    def _deserialize_additional_fields(self, data: dict, output: 'InferenceOutput'):
        """Hook for engine-specific deserialization logic"""
        return output

    def _serialize_additional_fields(self, output: 'InferenceOutput') -> dict:
        """Hook for engine-specific serialization logic"""
        return {}

@SerializerRegistry.register('vllm')
class VLLMSerializer(BaseSerializer):
    """Serializer for VLLM"""
    def _serialize_additional_fields(self, output: 'InferenceOutput') -> dict:
        """Hook for engine-specific serialization logic"""
        return {
            'response_logprobs': self._serialize_promptlogprobs(getattr(output, 'response_logprobs', None)),
            'raw_output': self._serialize_raw_output(getattr(output, 'raw_output', None))
        }

    def _deserialize_additional_fields(self, data: dict, output: 'InferenceOutput'):
        """Hook for engine-specific deserialization logic"""
        if 'response_logprobs' in data:
            setattr(output, 'response_logprobs', self._deserialize_promptlogprobs(data['response_logprobs']))
        if 'raw_output' in data:
            setattr(output, 'raw_output', self._deserialize_raw_output(data['raw_output']))
        return output

    def _serialize_promptlogprobs(self, logprobs: 'PromptLogprobs') -> List[dict] | None:
        """Serialize logprobs"""
        if logprobs is None:
            return None
        return [
            None if logprob_dict is None else {
                str(token_id): {
                    'logprob': info.logprob,
                    'rank': info.rank,
                    'decoded_token': info.decoded_token
                } for token_id, info in logprob_dict.items()
            }
            for logprob_dict in logprobs
        ]

    def _serialize_samplelogprobs(self, logprobs: 'SampleLogprobs') -> dict | None:
        """Serialize logprobs"""
        if logprobs is None:
            return None
        return {
            str(token_id): {
                'logprob': info.logprob,
                'rank': info.rank,
                'decoded_token': info.decoded_token
            }
            for token_id, info in logprobs.items()
        }

    def _serialize_raw_output(self, raw_output: 'RequestOutput') -> dict | None:
        """Serialize raw output"""
        if raw_output is None:
            return None
        return {
            'request_id': raw_output.request_id,
            'prompt': raw_output.prompt,
            'prompt_token_ids': raw_output.prompt_token_ids,
            'prompt_logprobs': self._serialize_promptlogprobs(getattr(raw_output, 'prompt_logprobs', None)),
            'outputs': [
                {
                    'index': output.index,
                    'text': output.text,
                    'token_ids': list(output.token_ids),
                    'cumulative_logprob': getattr(output, 'cumulative_logprob', None),
                    'logprobs': self._serialize_samplelogprobs(getattr(output, 'logprobs', None)),
                    'finish_reason': getattr(output, 'finish_reason', None),
                    'stop_reason': getattr(output, 'stop_reason', None),
                    'lora_request': self._serialize_lora_request(getattr(output, 'lora_request', None))
                }
                for output in raw_output.outputs
            ],
            'finished': raw_output.finished,
            'metrics': self._serialize_metrics(getattr(raw_output, 'metrics', None)),
            'lora_request': self._serialize_lora_request(getattr(raw_output, 'lora_request', None))
        }

    def _deserialize_promptlogprobs(self, data: dict) -> 'PromptLogprobs' | None:
        """Deserialize logprobs"""
        if data is None:
            return None
        return [
            None if logprob_dict is None else {
                int(token_id): Logprob(
                    logprob=info['logprob'],
                    rank=info['rank'],
                    decoded_token=info['decoded_token']
                )
                for token_id, info in logprob_dict.items()
            }
            for logprob_dict in data
        ]
    
    def _deserialize_samplelogprobs(self, data: dict) -> 'SampleLogprobs' | None:
        """Deserialize logprobs"""
        if data is None:
            return None
        return {
            int(token_id): Logprob(
                logprob=info['logprob'],
                rank=info['rank'],
                decoded_token=info['decoded_token']
            )
            for token_id, info in data.items()
        }

    def _deserialize_raw_output(self, data: dict) -> 'RequestOutput' | None:
        """Deserialize raw output"""
        if data is None:
            return None
        return RequestOutput(
            request_id=data['request_id'],
            prompt=data['prompt'],
            prompt_token_ids=data['prompt_token_ids'],
            prompt_logprobs=self._deserialize_promptlogprobs(data.get('prompt_logprobs')),
            outputs=self._deserialize_outputs(data['outputs']),
            finished=data['finished'],
            metrics=self._deserialize_metrics(data.get('metrics')),
            lora_request=self._deserialize_lora_request(data.get('lora_request'))
        )
    
    def _deserialize_outputs(self, data: List[dict]) -> List['CompletionOutput']:
        """Deserialize outputs"""
        if data is None:
            return None
        return [
            CompletionOutput(
                index=output['index'],
                text=output['text'],
                token_ids=tuple(output['token_ids']),
                cumulative_logprob=output.get('cumulative_logprob', None),
                logprobs=self._deserialize_samplelogprobs(output.get('logprobs', None)),
                finish_reason=output.get('finish_reason', None),
                stop_reason=output.get('stop_reason', None),
                lora_request=self._deserialize_lora_request(output.get('lora_request', None))
            )
            for output in data
        ]

    def _deserialize_metrics(self, metrics_data: List[dict]) -> List['RequestMetrics']:
        """Deserialize metrics"""
        if not metrics_data:
            return None
        return [
            RequestMetrics(
                arrival_time=metric['arrival_time'],
                last_token_time=metric['last_token_time'],
                first_scheduled_time=metric['first_scheduled_time'],
                first_token_time=metric['first_token_time'],
                time_in_queue=metric['time_in_queue'],
                finished_time=metric['finished_time']
            )
            for metric in metrics_data
        ]

    def _deserialize_lora_request(self, data: dict) -> 'LoRARequest':
        """Deserialize LoRARequest"""
        if not data:
            return None
        return LoRARequest(
            lora_name=data['lora_name'],
            lora_int_id=data['lora_int_id'],
            lora_path=data['lora_path'],
            lora_local_path=data['lora_local_path'],
            long_lora_max_len=data['long_lora_max_len']
        )

    def _serialize_metrics(self, metrics: List['RequestMetrics']) -> List[dict]:
        """Serialize metrics"""
        if not metrics:
            return None
        return [
            {
                'arrival_time': metric.arrival_time,
                'last_token_time': metric.last_token_time,
                'first_scheduled_time': getattr(metric, 'first_scheduled_time', None),
                'first_token_time': getattr(metric, 'first_token_time', None),
                'time_in_queue': getattr(metric, 'time_in_queue', None),
                'finished_time': getattr(metric, 'finished_time', None)
            }
            for metric in metrics
        ]

    def _serialize_lora_request(self, lora_request: 'LoRARequest') -> dict:
        """Serialize LoRARequest"""
        if not lora_request:
            return None
        return {
            'lora_name': lora_request.lora_name,
            'lora_int_id': lora_request.lora_int_id,
            'lora_path': lora_request.lora_path,
            'lora_local_path': getattr(lora_request, 'lora_local_path', None),
            'long_lora_max_len': getattr(lora_request, 'long_lora_max_len', None)
        }

# TODO: Add serializers for other backends


