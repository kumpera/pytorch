import abc
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import torch
import io
import torch.distributed as dist

from .metadata import (
    Metadata,
)

from .resharding import (
    create_default_local_plan,
    create_default_global_plan,
    create_default_metadata_only_plan,
    create_default_checkpoint_metadata,
    DefaultPlanner
)

from .storage import (
    StorageWriter,
    WriteItem,
    WriteResult,
    LocalPlan,
    Planner
)

from .utils import DistWrapper
from .api import CheckpointException

# -------------- private functions --------------

"""
Bad part of the current design:

1) We use List[Plan] with ranks implicitly encoded in the index.
    Should we use Dict[int, Plan] instead?

Do we need to include the source rank of the write request?
    It's implicit in the Plan object (maybe needs to be in Plan)

How would this handle the planner load balancing across ranks?
    It can handle both through rejection and stealing.

2)Write data is eagerly evaluated
The current model forces data to be eagerly evaluated which disables incremental
    serialization/transformation.

3) The way the planner and the storage layer attach data is different
    storage does it through storage_data fields
    planner does it through extending PlanningClasses

4) The only MD that can be sent back from ranks is through WriteResults
    Not great

FIXME:
    There's no way to pass ShardedTensor and global planner/storage data
    to the final MD payload. At least not with the default planner.
"""

def _create_metadata_from_local_state_dict(state_dict: Dict[str, Any]) -> Metadata:
    plan = create_default_metadata_only_plan(state_dict)
    results = [WriteResult.from_write_item(req, "") for req in plan.items]
    return create_default_checkpoint_metadata([results])

"""
New protocol:

prepare: (coordinator)
    storage.prepare() -> None 

local_plan: (all ranks)
    planner.local_plan -> storage.local_plan -> LocalPlan

global_plan (coordinator)
    planner.global_plan -> storage.global_plan -> GlobalPlan (list of LocalPlan)

finalize_plan: (all ranks)
    planner.merge_plan(local, global) -> storage.prepare_writes -> write requests

write: (all ranks)
    storage.write_data -> [WriteResult]

finish: (coordinator)
    planner.create_metadata([WriteResult]) -> Metadata
    storagw.finish(metadata) -> None

Notes:

Should storage.prepare() return something?
Lots of kinds of metadata hard/odd to ship to create_md:
    - per checkpoint data
    - per rank data (this could be items with different data but same FQN)
Should storage.finish() return something?

How to present the global plan?
    List of LocalPlan
    Dict of rank -> LocalPLan
    A GlobalPlan object (which we'd broadcast to all ranks?)


Load protocol::

plan: (all ranks)
    planner.prepare_read -> read requests


read: (all ranks)
    storage.read()


Notes:
    Should we pass Planners around or functions?

    Should we introduce a similar protocol to save:
        local plan -> global plan -> merge -> execute
    This could enable things like read
    

"""
def save_state_dict(
    state_dict: Dict[str, Any],
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Planner = None
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
    is_coordinator = no_dist or dist.get_rank(process_group) == coordinator_rank
    distW = DistWrapper(process_group, not no_dist, coordinator_rank)

    if planner is None:
        planner = DefaultPlanner()

    _ = distW.run_on_coordinator("prepare storage", storage_writer.prepare)

    def local_step():
        local_plan = planner.create_local_plan(state_dict, is_coordinator)
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan
    
    def global_step(all_local_plans):
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    local_plan, central_plan = distW.map_scatter("plan", local_step, global_step)

    def write_data():
        final_local_plan = planner.merge_plans(local_plan, central_plan)
        tensor_write_requests, bytes_write_requests = storage_writer.prepare_writes(
            state_dict,
            final_local_plan,
            planner.resolve_data
        )
        all_writes = storage_writer.write_data(final_local_plan.storage_data, tensor_write_requests, bytes_write_requests)

        all_writes.wait()
        return all_writes.value()

    def finish_checkpoint(all_results):
        metadata = planner.create_checkpoint_metadata(all_results)
        storage_writer.finish(metadata=metadata)
        # XXX do we return some sort of storage/planner result?
        return None

    _ = distW.map_reduce("write", write_data, finish_checkpoint)

