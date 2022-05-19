import io
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)


@dataclass
class TensorInfo:
    properties: TensorProperties
    size: torch.Size

@dataclass
class BytesIOProperties:
    length: int

@dataclass
class ShardStorageMetadata:
    # FIXME include a TensorProperties for the shard
    shard_metadata: ShardMetadata
    storage_data: Any = None
    planner_data: Any = None

# Metadata for each param.
@dataclass
class ShardedTensorStorageMetadata:
    # rename to properties
    info: TensorInfo

    # Storage info for each Shard. There's no ordering requirement for this list.
    shards: List[ShardStorageMetadata]

    storage_data: Any = None
    planner_data: Any = None


@dataclass
class TensorStorageMetadata:
    info: TensorInfo
    storage_data: Any = None
    planner_data: Any = None

@dataclass
class BytesStorageMetadata:
    bytes_properties: BytesIOProperties
    storage_data: Any = None
    planner_data: Any = None

TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]
STORAGE_TYPES = Union[ShardedTensorMetadata, TensorStorageMetadata, BytesStorageMetadata]
TENSOR_STORAGE_TYPES = Union[ShardStorageMetadata, TensorStorageMetadata]
STORAGE_ITEM_MD = Union[Tuple[ShardMetadata, TensorInfo], TensorInfo, BytesIOProperties]

@dataclass
class Metadata:
    # Keys are the same from the `state_dict` used.
    state_dict_metadata: Dict[str, STORAGE_TYPES]
