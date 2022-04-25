import io
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)

TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]

@dataclass
class ShardStorageMetadata:
    shard_metadata: ShardMetadata
    # Unique identifier for this particular Shard
    storage_key: str
    # Length in bytes for this shard
    length: int


# Metadata for each param.
@dataclass
class ShardedTensorStorageMetadata:
    # Metadata for the sharded tensor itself
    tensor_metadata: ShardedTensorMetadata
    # Storage info for each Shard. There's no ordering requirement for this list.
    storage_metadata: List[ShardStorageMetadata]


@dataclass
class TensorStorageMetadata:
    # Unique indentifier for this tensor
    storage_key: str

    # Lenght in bytes of this tensor
    length: int


@dataclass
class Metadata:
    # Metadata for the state dict.
    # This includes the MD for Tensors and ShardedTensors. ByteIO objects are skipped
    state_dict_metadata: Dict[str, Union[ShardedTensorStorageMetadata, TensorStorageMetadata]]

    def __getstate__(self) -> bytes:
        serialized = pickle.dumps(self.state_dict_metadata)
        return serialized

    def __setstate__(self, state: bytes) -> None:
        self.state_dict_metadata = pickle.loads(state)


@dataclass
class BytesWriteRequest:
    bytes: io.BytesIO
    storage_key: str


@dataclass
class BytesReadRequest:
    bytes: io.BytesIO
    storage_key: str


@dataclass
class TensorWriteRequest:
    tensor: torch.Tensor
    storage_key: str


@dataclass
class TensorReadRequest:
    tensor: torch.Tensor
    storage_key: str
    # offset and length w.r.t. to the storage identified by ``storage_key``
    offsets: Tuple[int, ...]
    lengths: Tuple[int, ...]
