import io
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
    ShardMetadata,
)

from .metadata import (
    Metadata,
    BytesWriteRequest,
    TensorStorageMetadata,
    ShardStorageMetadata,
    ShardedTensorStorageMetadata,
    TensorStorageMetadata,
    TensorWriteRequest,
)
from .resharding import prepare_sharded_tensor_write
from .storage_writer import StorageWriter

# -------------- private functions --------------
def _compute_tensor_md(fqn: str, tensor: Tensor) -> TensorStorageMetadata:
    tensor = tensor.detach()
    storage_size = tensor.nelement() * tensor.element_size()

    return TensorStorageMetadata(
        storage_key= fqn,
        length= storage_size,
    )


def _prepare(
    state_dict: Dict[str, Any], always_add_tensors: bool = False
) -> Tuple[Metadata, Dict[str, int], List[BytesWriteRequest], List[TensorWriteRequest]]:
    """
    Uses the state_dict to build three things.

    metadata: Metadata
        The metatdata discribing the tensor / sharded tensor.
        And it is storage meta data. See "../metadata.py" for detail

    size_for_storage_keys: Dict[str, int]
        Key is the storage key name, value is its size
        It can used to pre allocate the storage for parallel and non sequential writes.

    tensor_write_requests: List[TensorWriteRequest]
        List of tensor write requests that should br perfromed by the writer.

    bytes_write_requests: List[BytesWriteRequest]
        List of byte write requests that should br perfromed by the writer.

    Subclasses can optionally overwrite the implementation here,
    if the default does not meet its requirement.
    Args:
        state_dict: The state_dict to operate on
        always_add_tensors: Include non-sharded tensors even if rank != 0
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
                not always_add_tensors
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
            write_reqs, md = prepare_sharded_tensor_write(obj, fqn)
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
    This public function defined the default behavior to save a state_dict
    Notes
    1. This is a WIP, the state_dict save with different versions of the code might not be compatible.
    2. The caller needs to ensure the correctness of the state_dict

    Sample Code
    ```
        my_model = MyModule()
        optimizer = Adagrad(my_model.parameters())
        ...

        model_state_dict = my_model.state_dict()
        optim_state_dict = optimizer.state_dict()

        ...
        # torch.distributed does not assume the the correctness of the state_dict
        # the caller needs to ensure the correctness of the state_dict
        optim_state_dict = some_function_to_cleanup_optim_state_dict(optim_state_dict)
        ...

        fs_storage_writer = torch.distributed._checkpoint.FileSystemWriter("/checkpoint/1")
        torch.distributed._checkpoint.save_state_dict(
            state_dict=model_state_dict,
            storage_writer=fs_stroage_writer,
        )
        torch.distributed._checkpoint.save_state_dict(
            state_dict=optim_state_dict,
            storage_writer=fs_stroage_writer,
        )
        ...
    ```

    Args:
        state_dict (Dict[str, Any]) : A state_dict
        storage_writer (StorageWriter): An instance of storage writer that performance the writes.
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
