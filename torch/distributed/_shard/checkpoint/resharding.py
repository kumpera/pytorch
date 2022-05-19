from functools import partial
import io
from typing import List, Tuple, Dict, Any, Union, Callable, cast,Optional

import torch
from torch import Tensor

from torch.futures import Future

from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import (
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties

from .metadata import (
    BytesStorageMetadata,
    ShardStorageMetadata,
    ShardedTensorStorageMetadata,
    TensorStorageMetadata,
    TensorInfo,
    BytesIOProperties,
    Metadata,
    TENSOR_STORAGE_TYPES
)

from .storage import (
    LocalPlan,
    WriteItem,
    WriteResult,
    BytesWriteRequest,
    TensorReadRequest,
    TensorWriteRequest,
    Planner,
    STATE_DICT_TYPE,
    RESOLVE_DATA_TYPE,

    RESOLVE_TENSOR_TYPE,
    COPY_TENSOR_TYPE,
)

def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ShardMetadata, current_shard: ShardMetadata
) -> List[Tuple[int, int, int, int]]:
    """
    Return the overlapping region between saved_shard and current_shard.
    There returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
    narrows = []
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.shard_offsets,
            current_shard.shard_offsets,
            saved_shard.shard_sizes,
            current_shard.shard_sizes,
        )
    ):
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows

def _prepare_generic_tensor_read(
    fqn: str,
    checkpoint_shards: List[Tuple[ShardMetadata, TENSOR_STORAGE_TYPES]],
    local_shards: List[Shard],
    resolve_callback: Callable[[str, TENSOR_STORAGE_TYPES, Tensor], Tensor],
    copy_callback: Callable[[str, TENSOR_STORAGE_TYPES, Tensor, Tensor], Optional[Future[None]]],
) -> List[TensorReadRequest]:
        
    read_reqs = []
    # this is a naive quadratic algo that can be optimized later
    for shard in local_shards:
        # scan all mds looking for chunks
        for storage_md in checkpoint_shards:
            shard_md_from_storage = storage_md[0]

            # do they overlap?
            if not _check_shard_metadata_pair_overlap(
                shard.metadata, shard_md_from_storage
            ):
                continue

            target_tensor = shard.tensor.detach()
            # FIXME change this to use filesystem::tensor_narrow_n (move it to utils)
            offsets = []
            lengths = []
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=shard_md_from_storage, current_shard=shard.metadata
            ):
                # Note that we do NOT want to make any tensor copy.
                # all operation must be view only
                target_tensor = torch.narrow(
                    target_tensor, dim, offset_for_current_tensor, length
                )
                offsets.append(offset_for_saved_tensor)
                lengths.append(length)

            read_reqs.append(
                TensorReadRequest(
                    fqn=fqn,
                    meta=storage_md[1],
                    tensor=target_tensor,
                    offsets=offsets,
                    lengths=lengths,
                    resolve=partial(resolve_callback, fqn, storage_md[1]),
                    copy=partial(copy_callback, fqn, storage_md[1]),
                )
            )
    return read_reqs

def _sharded_tensor_props_for(sharded_tensor: ShardedTensor) -> TensorInfo:
    return TensorInfo(
        properties=sharded_tensor.metadata().tensor_properties,
        size=sharded_tensor.metadata().size,
    )

def _tensor_props_for(tensor: torch.Tensor) -> TensorProperties:
    return TensorProperties(
        dtype=tensor.dtype,
        layout=tensor.layout,
        requires_grad=tensor.requires_grad,
        memory_format=torch.contiguous_format,
        pin_memory=tensor.is_pinned
    )

def _bytes_prop_for(bytes: io.BytesIO) -> BytesIOProperties:
    return BytesIOProperties(length=-1)

def _create_for_shardmd(fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata):
    info = _sharded_tensor_props_for(sharded_tensor)
    return WriteItem(
        fqn=fqn,
        meta=(shard_md, info)
    )

def _create_for_shard(fqn: str, sharded_tensor: ShardedTensor, shard: Shard):
    info = _sharded_tensor_props_for(sharded_tensor)
    return WriteItem(
        fqn=fqn,
        meta=(shard.metadata, info)
    )

def _create_for_tensor(fqn: str, tensor: torch.Tensor):
    return WriteItem(
        fqn=fqn,
        meta=TensorInfo(_tensor_props_for(tensor), tensor.size())
    )

def _create_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        fqn=fqn,
        meta=_bytes_prop_for(bytes)
    )

def create_default_metadata_only_plan(state_dict: Dict[str, Any]):
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor):
            for shard_md in obj.metadata().shards_metadata:
                requests.append(_create_for_shardmd(fqn, obj, shard_md))
        elif isinstance(obj, Tensor):
            requests.append(_create_for_tensor(fqn, obj))
        else:
            requests.append(_create_for_bytesio(fqn, obj))
    return LocalPlan(requests, None)

def create_default_local_plan(state_dict: Dict[str, Any], is_coordinator: bool):
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor):
            for shard in obj.local_shards():
                requests.append(_create_for_shard(fqn, obj, shard))
        elif isinstance(obj, Tensor):
            if is_coordinator:
                requests.append(_create_for_tensor(fqn, obj))
        else:
            if is_coordinator:
                # FIXME do we need to provide some size guestimate at this stage?
                requests.append(_create_for_bytesio(fqn, obj))
    return LocalPlan(requests, None)

# Default global plan does nothing
def create_default_global_plan(all_plans: List[LocalPlan]) -> List[LocalPlan]:
    return all_plans


# FIXME this stage triggers serialization of blobs, make it plugable
def default_prepare_writes(
    state_dict: STATE_DICT_TYPE, 
    plan: LocalPlan, 
    resolve_callback: RESOLVE_DATA_TYPE
) -> Tuple[List[TensorWriteRequest], List[BytesWriteRequest]]:

    def resolve_bytes(wi):
        bytes = resolve_callback(state_dict, wi)
        cast(BytesIOProperties, wi.meta).length = len(bytes.getbuffer())
        return bytes

    def resolve_tensor(wi):
        tensor = resolve_callback(state_dict, wi)
        return tensor

    tensor_writes = []
    bytes_writes = []
    for wi in plan.items:
        if wi.is_bytesio:
            bytes_writes.append(BytesWriteRequest(wi, resolve_bytes))
        else:
            tensor_writes.append(TensorWriteRequest(wi, resolve_tensor))

    return (tensor_writes, bytes_writes)

def create_default_checkpoint_metadata(results: List[Union[BaseException, List[WriteResult]]]) -> Metadata:
    md = dict()

    for wr_list in results:
        for wr in wr_list:
            if isinstance(wr.meta, Tuple):
                shard_info, tinfo = wr.meta
                container = md.get(wr.fqn, None)
                if container is None:
                    container = ShardedTensorStorageMetadata(
                        info=tinfo,
                        shards=[],
                    )
                    md[wr.fqn] = container

                container.shards.append(ShardStorageMetadata(
                    shard_metadata=shard_info,
                    storage_data=wr.storage_data,
                    planner_data=wr.planner_data,
                ))
            elif isinstance(wr.meta, TensorInfo):
                assert wr.fqn not in md

                md[wr.fqn] = TensorStorageMetadata(
                    info=wr.meta,
                    storage_data=wr.storage_data,
                    planner_data=wr.planner_data,
                )
            else:
                assert wr.fqn not in md
                md[wr.fqn] = BytesStorageMetadata(
                    bytes_properties=wr.meta,
                    storage_data=wr.storage_data,
                    planner_data=wr.planner_data,
                )

    return Metadata(state_dict_metadata=md)

class DefaultPlanner(Planner):
    def create_local_plan(self, state_dict: Dict[str, Any], is_coordinator: bool) -> LocalPlan:
        return create_default_local_plan(state_dict, is_coordinator)

    def create_global_plan(self, all_plans: List[LocalPlan]) -> List[LocalPlan]:
        return create_default_global_plan(all_plans)

    def merge_plans(self, original_plan: LocalPlan, new_plan: LocalPlan) -> LocalPlan:
        return new_plan

    def create_checkpoint_metadata(self, all_results: List[Union[BaseException, List[WriteResult]]]) -> Metadata:
        return create_default_checkpoint_metadata(all_results)

    def resolve_data(self, state_dict: Dict[str, Any], write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        obj = write_item.lookup(state_dict)
        if write_item.is_bytesio:
            bytes = io.BytesIO()
            torch.save(obj, bytes)
            obj = bytes
        return obj