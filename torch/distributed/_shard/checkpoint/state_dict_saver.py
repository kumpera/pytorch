import abc
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import torch
import io
import torch.distributed as dist

from .resharding import (
    create_default_global_plan,
    create_default_metadata_only_plan,
    DefaultSavePlanner
)

from .storage import (
    StorageWriter,
    SavePlanner
)

from .metadata import Metadata
from .utils import _DistWrapper

def _create_metadata_from_local_state_dict(state_dict: Dict[str, Any]) -> Metadata:
    plan = create_default_metadata_only_plan(state_dict)
    _, md = create_default_global_plan([plan])
    return md


def save_state_dict(
    state_dict: Dict[str, Any],
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: SavePlanner = None
) -> Metadata:
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

    If using the `process_group` argument, make sure that only its ranks
    call `save_state_dict` and that all data in state_dict belong to it.

    This function can be used to save a state_dict with an intialized process
    group by passing ``no_dist=True``. This can be used to produce a checkpoint
    that can consumed by load_state_dict is a SPMD fashion.

    Args:
        state_dict (Dict[str, Any]) : A state_dict
        storage_writer (StorageWriter): Instance of StorageWrite use to perform writes.
        process_group (ProcessGroup): ProcessGroup to be used for cross-rank synchronization
        coordinator_rank (int): Rank to use to coordinate the checkpoint, rank0 is used by default
        no_dist (bool): Don't attempt to save in SPMD style. Default to False

    Example:
        >>> my_model = MyModule()
        >>> # We must call this function prior to state_dict()
        >>> my_model._register_state_dict_hook(state_dict_hook)

        >>> model_state_dict = my_model.state_dict()

        >>> fs_storage_writer = torch.distributed._shard.checkpoint.FileSystemWriter("/checkpoint/1")
        >>> torch.distributed._shard.checkpoint.save_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_writer=fs_stroage_writer,
        >>> )

    .. note:: save_state_dict uses collectives to coordinate writes across ranks.
        For NCCL-based process groups, internal tensor representations of objects
        must be moved to the GPU device before communication takes place. In this
        case, the device used is given by ``torch.cuda.current_device()`` and it
        is the user's responsibility to ensure that this is set so that each rank
        has an individual GPU, via ``torch.cuda.set_device()``
    """
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()

    def local_step():
        planner.init(state_dict, distW.is_coordinator)
        storage_writer.init(distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan = distW.reduce_scatter("plan", local_step, global_step)

    def write_data():
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        return all_writes.value()

    def finish_checkpoint(all_results):
        metadata = planner.create_checkpoint_metadata(all_results)
        storage_writer.finish(metadata=metadata, results=all_results)
        return metadata

    return distW.all_reduce("write", write_data, finish_checkpoint)

