
import io
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata
)
from torch.distributed._shard.sharding_spec._internals import (
    validate_non_overlapping_shards_metadata,
    _check_shard_metadata_pair_overlap,
)

from .metadata import (
    BytesReadRequest,
    TensorReadRequest,
    Metadata,
    ShardedTensorStorageMetadata,
    TensorStorageMetadata,
)
from .resharding import (
    prepare_sharded_tensor_read,
    _shards_get_overlap_region_wrt_saved_tensor
)
from .storage_reader import StorageReader


def _reshard_and_prepare_read_request(
    state_dict: Dict[str, Any], metadata_from_storage: Metadata
) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensors

    NOTE:
    During the save,
    """
    tensor_read_requests = []
    bytes_read_requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach()
            storage_size = tensor.nelement() * tensor.element_size()

            rr = TensorReadRequest(
                tensor=tensor,
                storage_key=fqn,
                offsets=tuple([0] * len(tensor.size())),
                lengths=tensor.size(),
            )

            tensor_read_requests.append(rr)
        elif isinstance(obj, ShardedTensor):
            md = metadata_from_storage.state_dict_metadata[fqn]
            tensor_read_requests += prepare_sharded_tensor_read(md, obj)
        else:
            # This is actually hard to handle correctly
            # If the value is not a tensor but any random obj,
            # we cannot just write whatever memory it points to inplace
            # the best we can to is to replace it with an object of the same type
            bytes_io = io.BytesIO()
            brr = BytesReadRequest(
                bytes=bytes_io,
                storage_key=fqn,
            )
            bytes_read_requests.append(brr)

    return (bytes_read_requests, tensor_read_requests)


def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
) -> None:
    """
    This public function defines the default behavior to load a state_dict

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

        fs_storage_loader = torch.distributed._checkpoint.FileSystemLoader("/checkpoint/1")
        torch.distributed._checkpoint.load_state_dict(
            state_dict=model_state_dict,
            storage_reader=fs_stroage_loader,
        )
        torch.distributed._checkpoint.load_state_dict(
            state_dict=optim_state_dict,
            storage_reader=fs_stroage_loader,
        )

        # module.load_state_dict() functon might have customized steps
        # to flush the state_dict, must call them to
        # ensure the correct behavior
        my_model.load_state_dict(model_state_dict)
        optim_state_dict.load_state_dict(optim_state_dict)
        ...
    ```
    Args:
        state_dict (Dict[str, Any]) : A state_dict to load to. Note that this
            state dict will updated in places.
        storage_reader (StorageReader): An instance of storage loader.
    """

    metadata = storage_reader.read_metadata()
    bytes_read_requests, tensor_read_requests = _reshard_and_prepare_read_request(
        state_dict=state_dict, metadata_from_storage=metadata
    )
    bytes_futures = storage_reader.read_bytes(bytes_read_requests)
    tensor_futures = storage_reader.read_tensors(tensor_read_requests)
    bytes_futures.wait()

    # Addtional steps are required to convert the bytes to its original type
    # Note that this is NOT inplace,
    # it creating a new object and replace what's in the state dict
    for req in bytes_read_requests:
        fqn = req.storage_key
        # Ensure the BytesIO is rewound
        req.bytes.seek(0)
        state_dict[fqn] = torch.load(req.bytes)

    tensor_futures.wait()


def _validate_sharded_tensor(
    tensor_md: ShardedTensorMetadata, checkpoint_md: ShardedTensorMetadata
) -> List[str]:
    # We assume the incoming tensor has being validated during construction

    res = []
    # To ensure a checkpoint can satisfy loading a ST, we compute the loading
    # plans for all shards and see if they are doable.
    try:
        # this API returns a list of issues
        validate_non_overlapping_shards_metadata(
            checkpoint_md.tensor_metadata.shards_metadata
        )
    except ValueError as e:
        res.append(str(e))

    for shard_md in tensor_md.shards_metadata:
        read_volume = 0
        for storage_md in checkpoint_md.storage_metadata:
            shard_md_from_storage = storage_md.shard_metadata
            assert shard_md_from_storage is not None
            # this is a naive quadratic algo that can later be optimized by
            #   sorting metadata and the shards md

            # do they overlap?
            if not _check_shard_metadata_pair_overlap(shard_md, shard_md_from_storage):
                continue

            shard_volume = 1
            for (_, _, _, length,) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=shard_md_from_storage, current_shard=shard_md
            ):
                shard_volume *= length
            read_volume += shard_volume

        shard_volume = 1
        for size in shard_md.shard_sizes:
            shard_volume *= size
        if read_volume != shard_volume:
            res.append(
                f"Shard {shard_md} only has {read_volume} available" +
                "elements but needs {shard_volume}"
            )
    return res


def validate_metadata(
    state_dict: Dict[str, Any], metadata: Metadata
) -> Optional[List[str]]:
    """
    Verify if it's possible to correctly load `state_dict` from `metadata`.

    This method can be used to validate if a checkpoint is usable with a given model.

    Sample Code
    ```
        my_model: torch.nn.Model = ....
        my_reader: torch.distributed._checkpoint.StorageReader = ...

        res = torch.distributed._checkpoint.validate_metadata(my_model.state_dict(), my_reader.read_metadata())
    ```
    Args:
        state_dict: A state_dict to verify if it's loadable.
        metadata: Checkpoint metadata to verify against.

    Returns:
        None if no issue was found or a List[str] of issues.
    """
    res = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, torch.Tensor):
            if fqn not in metadata.state_dict_metadata:
                res.append(f"{fqn}: Could not find Tensor metadata")
                continue
            md = metadata.state_dict_metadata[fqn]
            if not isinstance(md, TensorStorageMetadata):
                res.append(f"{fqn}: Expected TensorStorageMetadata but found: {type(md)}")
                continue
            md_size = md.length
            tensor_size = obj.numel() * obj.element_size()
            if md_size != tensor_size:
                res.append(
                    f"{fqn}: Incompatible tensor size: expected {tensor_size} but found {md_size}"
                )
        elif isinstance(obj, ShardedTensor):
            if fqn not in metadata.state_dict_metadata:
                res.append(f"{fqn}: Could not find ShardedTensor metadata")
                continue
            md = metadata.state_dict_metadata[fqn]
            if not isinstance(md, ShardedTensorStorageMetadata):
                res.append(f"{fqn}: Expected ShardedTensorStorageMetadata but found: {type(md)}")
                continue
            # Check if the overall ShardedTensor size is the same. Individual shards don't matter as we can reshard.
            md_size = list(md.tensor_metadata.size)
            tensor_size = list(obj.metadata().size)
            if md_size != tensor_size:
                res.append(
                    f"{fqn}: Incompatible ShardedTensor size: expectected {tensor_size} but found {md_size}"
                )
            res += _validate_sharded_tensor(obj.metadata(), md)

    return res if len(res) > 0 else None
