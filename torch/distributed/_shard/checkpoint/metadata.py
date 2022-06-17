from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties

import torch

@dataclass(frozen=True, unsafe_hash=True)
class MetadataIndex:
    """
    This can be used to index into any item in Metadata
    """
    fqn: str
    chunk_index: Optional[int]

@dataclass
class TensorInfo:
    properties: TensorProperties
    size: torch.Size


@dataclass
class ChunkStorageMetadata:
    """
    Each chunk is expected to have the same properties of the TensorStorageMetadata that includes it.
    """
    offsets: torch.Size
    sizes: torch.Size

    # This is reported by the storage layer so it might include serialization overhead
    size_in_bytes: int

@dataclass
class TensorStorageMetadata:
    properties: TensorInfo

    chunks: List[ChunkStorageMetadata]

@dataclass
class BytesStorageMetadata:
    size_in_bytes: int

STORAGE_TYPES = Union[TensorStorageMetadata, BytesStorageMetadata]

@dataclass
class Metadata:
    # Keys are the same from the `state_dict` used.
    state_dict_metadata: Dict[str, STORAGE_TYPES]
    storage_data: Any = None
    planner_data: Any = None
