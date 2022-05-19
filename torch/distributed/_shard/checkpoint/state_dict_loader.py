from functools import partial
import traceback
import io
from typing import Any, Dict, List, Tuple, Optional, cast
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata
)
from torch.distributed._shard.sharding_spec._internals import (
    validate_non_overlapping_shards_metadata,
    _check_shard_metadata_pair_overlap,
)

from .metadata import (
    BytesStorageMetadata,
    Metadata,
    ShardedTensorStorageMetadata,
    ShardStorageMetadata,
    TensorStorageMetadata,
)
from .resharding import (
    DefaultPlanner,
    _prepare_generic_tensor_read,
    _shards_get_overlap_region_wrt_saved_tensor
)
from .storage import (
    BytesReadRequest,
    StorageReader,
    TensorReadRequest,
    Planner,
)
from .api import CheckpointException


def _create_shard_metadata(size: torch.Size) -> ShardMetadata:
    return ShardMetadata(
        shard_offsets=[0] * len(size),
        shard_sizes=list(size),
    )

def _create_shard_for(tensor: Tensor) -> Shard:
    return Shard(
        tensor=tensor,
        metadata=_create_shard_metadata(tensor.size()),
    )

def _create_checkpoint_shard_for(storage: TensorStorageMetadata) -> ShardStorageMetadata:
    return ShardStorageMetadata(
        # The metadata device is not used during loading.
        shard_metadata=_create_shard_metadata(storage.size),
        storage_key=storage.storage_key,
    )

class LoadPlanner:
    def load_bytes(self, state_dict, fqn, md, stream):
        state_dict[fqn] = torch.load(stream)

    def resolve_tensor(self, fqn, md, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def copy_tensor(self, fqn, md, dest: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        dest.copy_(src)

def _reshard_and_prepare_read_request(
    state_dict: Dict[str, Any],
    metadata_from_storage: Metadata,
    planner: LoadPlanner
) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensor
    """
    tensor_read_requests = []
    bytes_read_requests = []
    for fqn, obj in state_dict.items():
        md = metadata_from_storage.state_dict_metadata[fqn]
        if isinstance(obj, ShardedTensor):
            local_shards = obj.local_shards()
        elif isinstance(obj, torch.Tensor):
            tensor = obj.detach()
            local_shards = [_create_shard_for(tensor)]
        else:
            if isinstance(md, BytesStorageMetadata):
                byte_req = BytesReadRequest(
                    fqn=fqn,
                    meta=md,
                    copy=partial(planner.load_bytes, state_dict, fqn, md)
                )
                bytes_read_requests.append(byte_req)
            else:
                raise ValueError(
                    f"Invalid checkpoint metadata for {fqn}, " +
                    f"expected BytesStorageMetadata but found {type(md)}"
                )
            continue

        if isinstance(md, ShardedTensorStorageMetadata):
            checkpoint_shards = [
                (smd.shard_metadata, smd) for smd in md.shards
            ]
        elif isinstance(md, TensorStorageMetadata):
            checkpoint_shards = [
                (_create_shard_metadata(md.info.size, "cpu"), md)    
            ]
        else:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, " +
                f"expected TensorStorageMetadata but found {type(md)}"
            )


        tensor_read_requests += _prepare_generic_tensor_read(
            fqn,
            checkpoint_shards,
            local_shards,
            planner.resolve_tensor,
            planner.copy_tensor)

    return (bytes_read_requests, tensor_read_requests)

def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: LoadPlanner = None
) -> None:
    """
    Load a distributed state_dict in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`.

    When loading ShardedTensor instances, each rank only
    reads data for their local shards.

    All tensors in ``state_dict`` must be allocated on their
    destination device prior to calling this function.

    All non-tensor data is loaded using `torch.load()` and modified in place
    on state_dict.

    Users must call `load_state_dict` on the root module to ensure load
    pos-processing and non-tensor data properly propagates.

    This function can be used for local inference and load a checkpoint
    produced by ``save_state_dict`` without having a process group initialized
    by passing ``no_dist=True`` and by using Tensors instead of ShardedTensors.

    Args:
        state_dict (Dict[str, Any]) : The state_dict to load. Note that this
            state dict will updated in places.
        storage_reader (StorageReader): StorageReader used to load data from.
        process_group (ProcessGroup): ProcessGroup to be used for cross-rank synchronization
        coordinator_rank (int): Rank to use to coordinate the checkpoint, rank0 is used by default
        no_dist (bool): Don't attempt to load in SPMD style. Default to False

    Returns:
        None.

    Examples
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_loader = torch.distributed._shard.checkpoint.FileSystemLoader("/checkpoint/1")

        >>> torch.distributed._shard.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_loader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)

    .. note:: load_state_dict uses collectives to coordinate reads across ranks.
        For NCCL-based process groups, internal tensor representations of objects
        must be moved to the GPU device before communication takes place. In this
        case, the device used is given by ``torch.cuda.current_device()`` and it
        is the user's responsibility to ensure that this is set so that each rank
        has an individual GPU, via ``torch.cuda.set_device()``
    """
    is_coordinator = no_dist or dist.get_rank(process_group) == coordinator_rank
    if planner is None:
        planner = LoadPlanner()

    try:
        metadata = storage_reader.read_metadata()
        # load_plan = planner.create_load_plan(state_dict, metadata)

        bytes_read_requests, tensor_read_requests = _reshard_and_prepare_read_request(
            state_dict=state_dict,
            metadata_from_storage=metadata,
            planner=planner
        )
        bytes_futures = storage_reader.read_bytes(bytes_read_requests)
        tensor_futures = storage_reader.read_tensors(tensor_read_requests)

        bytes_futures.wait()
        tensor_futures.wait()
        result = None
    except BaseException as e:
        traceback.print_exc()
        result = e

    global_result: Optional[CheckpointException] = None
    if not no_dist:
        all_errors = [None] * dist.get_world_size(process_group)

        dist.all_gather_object(
            object_list=all_errors,
            obj=result,
            group=process_group)

        node_failures = cast(Dict[int, BaseException], {i: err for i, err in enumerate(all_errors) if err is not None})
        if len(node_failures) > 0:
            global_result = CheckpointException("failed to read checkpoint", node_failures)
    elif result is not None:
        global_result = CheckpointException("failed to read storage", {coordinator_rank : result})

    if global_result is not None:
        raise global_result


def _validate_sharded_tensor(
    tensor_md: ShardedTensorMetadata, checkpoint_md: ShardedTensorStorageMetadata
) -> None:
    # We assume the incoming tensor has being validated during construction

    # To ensure a checkpoint can satisfy loading a ST, we compute the loading
    # plans for all shards and see if they are doable.
    validate_non_overlapping_shards_metadata(
        [s.shard_metadata for s in checkpoint_md.shards]
    )

    for shard_md in tensor_md.shards_metadata:
        read_volume = 0
        for storage_md in checkpoint_md.shards:
            shard_md_from_storage = storage_md.shard_metadata

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
            raise ValueError(
                f"Shard {shard_md} only has {read_volume} available" +
                f" elements but needs {shard_volume}"
            )

def validate_metadata(
    state_dict: Dict[str, Any], metadata: Metadata
) -> None:
    """
    Verify if it's possible to correctly load `state_dict` from `metadata`.

    This method validate if a checkpoint is usable with a given model
    state_dict without loading it. It will raise ValueError if it finds
    anything problematic.

    Args:
        state_dict: A state_dict to verify if it's loadable.
        metadata: Checkpoint metadata to verify against.

    Returns:
        None

    Example:
        >>> my_model: torch.nn.Model = ....
        >>> my_reader: torch.distributed._shard.checkpoint.StorageReader = ...

        >>> torch.distributed._shard.checkpoint.validate_metadata(my_model.state_dict(), my_reader.read_metadata())
        None
    ```

    """
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor):
            if fqn not in metadata.state_dict_metadata:
                raise ValueError(f"{fqn}: Could not find ShardedTensor metadata")

            md = metadata.state_dict_metadata[fqn]
            if not isinstance(md, ShardedTensorStorageMetadata):
                raise ValueError(f"{fqn}: Expected ShardedTensorStorageMetadata but found: {type(md)}")

            # Check if the overall ShardedTensor size is the same. Individual shards don't matter as we can reshard.
            md_size = md.size
            tensor_size = obj.metadata().size
            if md_size != tensor_size:
                raise ValueError(
                    f"{fqn}: Incompatible ShardedTensor size: expectected {tensor_size} but found {md_size}"
                )

            _validate_sharded_tensor(obj.metadata(), md)
        elif isinstance(obj, torch.Tensor):
            if fqn not in metadata.state_dict_metadata:
                raise ValueError(f"{fqn}: Could not find Tensor metadata")

            md = metadata.state_dict_metadata[fqn]
            if not isinstance(md, TensorStorageMetadata):
                raise ValueError(f"{fqn}: Expected TensorStorageMetadata but found: {type(md)}")

            if md.size != obj.size():
                raise ValueError(
                    f"{fqn}: Incompatible tensor size: expected {obj.size()} but found {md.size}"
                )
