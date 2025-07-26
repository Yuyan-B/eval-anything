import gc
import os
from typing import Tuple, Union
import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
def get_current_device() -> 'torch.device':
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = 'xpu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_npu_available():
        device = 'npu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_mps_available():
        device = 'mps:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    else:
        device = 'cpu'

    return torch.device(device)
def set_device(device_id) -> str:
    r"""
    Sets the device.
    """
    if is_torch_xpu_available():
        device = f'xpu:{device_id}'
    elif is_torch_npu_available():
        device = f'npu:{device_id}'
    elif is_torch_mps_available():
        device = f'mps:{device_id}'
    elif is_torch_cuda_available():
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'
    return device
def torch_set_device(device: Union[torch.device, str, int, None]) -> None:
    r"""
    Sets the device for PyTorch.
    """
    if is_torch_npu_available():
        torch.npu.set_device(device)
    elif is_torch_cuda_available():
        torch.cuda.set_device(device)
def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0