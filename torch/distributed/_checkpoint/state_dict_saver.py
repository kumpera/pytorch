import io
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)

from .metadata import (
    Metadata,
    BytesWriteRequest,
    TensorStorageMetadata,
    TensorWriteRequest,
)
from .resharding import _prepare_sharded_tensor_write
from .storage_writer import StorageWriter

# -------------- private functions --------------
def _compute_tensor_md(fqn: str, tensor: Tensor) -> TensorStorageMetadata:
    return TensorStorageMetadata(
        storage_key=fqn,
        size=tensor.size()
    )


def _prepare(
    state_dict: Dict[str, Any], include_non_replicated_tensors: bool = False
) -> Tuple[Metadata, Dict[str, int], List[BytesWriteRequest], List[TensorWriteRequest]]:
    """
    Build a serialization plan for a given state_dict
    By default, regular tensors are only included if execution from `rank 0`.

    Args:
        state_dict: The instance to plan for.
        include_non_replicated_tensors: Include non-sharded tensors even if rank != 0

    Returns:
        A tuple with the following values:

        metadata: Metadata
        The storage metadata describing Tensor and ShardedTensors
        instances found in `state_dict`. See `Metadata` for the schema.

        size_for_storage_keys: Dict[str, int]
            Key is the storage key name, value is the associated size
            It can used to pre allocate the storage for parallel and non sequential writes.

        bytes_write_requests: List[BytesWriteRequest]
            List of ByteIO write requests that should be performed by the writer.

        tensor_write_requests: List[TensorWriteRequest]
            List of Tensor write requests that should be performed by the writer.

    NB: `include_non_replicated_tensors=True` only makes sense for testing or validation.
    Do not use it with load/save as it will lead to invalid checkpoints.

    """
    metadata = Metadata(state_dict_metadata={})
    tensor_write_requests: List[TensorWriteRequest] = []
    bytes_write_requests: List[BytesWriteRequest] = []

    for fqn, obj in state_dict.items():
        if isinstance(obj, Tensor):
            # The assumption is that non ShardedTensors are full replicated across all ranks
            # So we just need one from Rank 0.
            # If that's not the case, we will update later.
            if (
                not include_non_replicated_tensors
                and dist.is_initialized()
                and dist.get_rank() != 0
            ):
                pass
            else:
                tensor_write_requests.append(
                    TensorWriteRequest(
                        tensor=obj.detach(),
                        storage_key=fqn,
                    )
                )
                metadata.state_dict_metadata[fqn] = _compute_tensor_md(fqn, obj)
        elif isinstance(obj, ShardedTensor):
            write_reqs, md = _prepare_sharded_tensor_write(obj, fqn)
            tensor_write_requests += write_reqs
            metadata.state_dict_metadata[fqn] = md
        else:
            bytes_io = io.BytesIO()
            torch.save(obj, bytes_io)
            bwr = BytesWriteRequest(
                bytes=bytes_io,
                storage_key=fqn,
            )
            bytes_write_requests.append(bwr)

    storage_keys: Dict[str, int] = {
        req.storage_key: req.tensor.nelement() * req.tensor.element_size()
        for req in tensor_write_requests
    }

    return (metadata, storage_keys, bytes_write_requests, tensor_write_requests)


def save_state_dict(
    state_dict: Dict[str, Any],
    storage_writer: StorageWriter,
) -> None:
    """
    Save a distributed model in SPMD style.

    This function is different from ``torch.save()`` as it handles
    ``ShardedTensor`` by having each rank only save their local shards.

    To produce a state_dict with ShardedTensor instances you must call
    ``_register_state_dict_hook`` on the top module with value
    `torch.distributed._shard.sharded_tensor.state_dict_hook` prior to
    calling `state_dict()` on the top module.

    There is no guarantees of Backwards Compatibility across PyTorch versions
    for saved state_dicts.

    Args:
        state_dict (Dict[str, Any]) : A state_dict
        storage_writer (StorageWriter): Instance of StorageWrite use to perform writes.

    Example:
        >>> my_model = MyModule()
        >>> # We must call this function prior to state_dict()
        >>> my_model._register_state_dict_hook(state_dict_hook)

        >>> model_state_dict = my_model.state_dict()

        >>> fs_storage_writer = torch.distributed._checkpoint.FileSystemWriter("/checkpoint/1")
        >>> torch.distributed._checkpoint.save_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_writer=fs_stroage_writer,
        >>> )
    """
    (
        metadata,
        storage_keys,
        bytes_write_requests,
        tensor_write_requests,
    ) = _prepare(state_dict)
    storage_writer.prepare_storage(storage_keys=storage_keys)
    storage_writer.write_metadata(metadata=metadata)
    bytes_futures = storage_writer.write_bytes(bytes_write_requests)
    tensor_futures = storage_writer.write_tensors(tensor_write_requests)
    bytes_futures.wait()
    tensor_futures.wait()
