import io
from multiprocessing.sharedctypes import Value
from typing import List, Tuple, Dict, Any, Union, cast

import torch
from torch import Tensor

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
    ChunkStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
    TensorInfo,
    Metadata,
    STORAGE_TYPES,
)

from .storage import (
    SavePlan,
    WriteItem,
    WriteItemType,
    SavePlanner,
    WriteResult,

    LoadPlan,
    ReadItem,
    LoadPlanner,
    STATE_DICT_TYPE,
)

from .utils import tensor_narrow_n

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
        pin_memory=tensor.is_pinned()
    )


def _chunk_for_sharmd(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes),
        size_in_bytes=-1,
    )

def _create_for_shardmd(fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> WriteItem:
    return WriteItem(
        fqn=fqn,
        type=WriteItemType.SHARD,
        chunk=_chunk_for_sharmd(shard_md),
        tensor_info=_sharded_tensor_props_for(sharded_tensor),
    )

def _create_for_shard(fqn: str, sharded_tensor: ShardedTensor, shard: Shard) -> WriteItem:
    return WriteItem(
        fqn=fqn,
        type=WriteItemType.SHARD,
        chunk=ChunkStorageMetadata(
            offsets=torch.Size(shard.metadata.shard_offsets),
            sizes=torch.Size(shard.metadata.shard_sizes),
            size_in_bytes=-1,
        ),
        tensor_info=_sharded_tensor_props_for(sharded_tensor),
    )

def _create_for_tensor(fqn: str, tensor: torch.Tensor) -> WriteItem:
    return WriteItem(
        fqn=fqn,
        type=WriteItemType.TENSOR,
        chunk=ChunkStorageMetadata(
            offsets=torch.Size([0] * len(tensor.size())),
            sizes=tensor.size(),
            size_in_bytes=-1,
        ),
        tensor_info=TensorInfo(_tensor_props_for(tensor), tensor.size()),
    )

def _create_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        fqn=fqn,
        type=WriteItemType.BYTE_IO,
    )

def create_default_metadata_only_plan(state_dict: Dict[str, Any]) -> SavePlan:
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor):
            for shard_md in obj.metadata().shards_metadata:
                requests.append(_create_for_shardmd(fqn, obj, shard_md))
        elif isinstance(obj, Tensor):
            requests.append(_create_for_tensor(fqn, obj))
        else:
            requests.append(_create_for_bytesio(fqn, obj))
    return SavePlan(requests)

def create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    if isinstance(object, ShardedTensor):
        return [_create_for_shard(fqn, object, shard) for shard in object.local_shards()]
    elif isinstance(object, Tensor):
        return [_create_for_tensor(fqn, object)]
    else:
       return [_create_for_bytesio(fqn, object)]

def create_default_local_plan(state_dict: Dict[str, Any], is_coordinator: bool):
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor) or is_coordinator:
            requests += create_write_items(fqn, obj)
    return SavePlan(requests)

def create_default_global_plan(all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    """
    The default plan creates a Metadata object with -1 as size_in_bytes
    """
    md: Dict[str, STORAGE_TYPES]= dict()

    for plan in all_plans:
        for item in plan.items:
            if not item.is_shard:
                assert item.fqn not in md

            if item.is_bytesio:
                md[item.fqn] = BytesStorageMetadata(size_in_bytes=-1)
            else:
                assert item.tensor_info is not None
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(item.fqn, TensorStorageMetadata(
                        properties=item.tensor_info,
                        chunks=[],
                    ))
                )

                assert item.chunk is not None, f"Cannot create MD for tensor without bounds. FQN: {item.fqn}"
                tensor_md.chunks.append(item.chunk)

    return (all_plans, Metadata(md))

def get_chunk_index(list: List[ChunkStorageMetadata], offset: torch.Size):
    for i,c in enumerate(list):
        if c.offsets == offset:
            return i
    raise ValueError(f"Offset {offset} not found")

def populate_metadata_with_write_results(md: Metadata, results: List[List[WriteResult]]) -> None:
    """
    By default we populate the following:
        size_in_bytes of all leaf items
        md::planner_data with a Dict[MetadataIndex, Any] by aggregating over WriteResults
        md::storage_data with a Dict[MetadataIndex, Any] by aggregating over WriteResults
    """

    for wr_list in results:
        for wr in wr_list:
            item = md.state_dict_metadata[wr.fqn]
            if isinstance(item, TensorStorageMetadata):
                item.chunks[get_chunk_index(item.chunks, wr.chunk_offset)].size_in_bytes = wr.size_in_bytes
            else:
                item.size_in_bytes = wr.size_in_bytes

def _create_shard_metadata(size: torch.Size) -> ShardMetadata:
    return ShardMetadata(
        shard_offsets=[0] * len(size),
        shard_sizes=list(size),
    )

def _create_shard_for(tensor: Tensor) -> Shard:
    return Shard(
        tensor=tensor,
        metadata=_create_shard_metadata(tensor.size())
    )

def _get_extra_metadata(md: Metadata, key: MetadataIndex):
    planner_data = None
    if isinstance(md.planner_data, Dict):
        planner_data = md.planner_data.get(key, None)
    storage_data = None
    if isinstance(md.storage_data, Dict):
        storage_data = md.storage_data.get(key, None)
    return (planner_data, storage_data)


def _create_sharded_read_items(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_shards: List[Shard],
) -> List[ReadItem]:

    read_items = []
    # this is a naive quadratic algo that can be optimized later
    for shard in local_shards:
        for idx, storage_md in enumerate(checkpoint_md.chunks):
            shard_md_from_storage = ShardMetadata(
                shard_sizes=list(storage_md.sizes),
                shard_offsets=list(storage_md.offsets),
            )

            if not _check_shard_metadata_pair_overlap(
                shard.metadata, shard_md_from_storage
            ):
                continue

            storage_offsets = []
            dest_offsets = []
            lengths = []
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=shard_md_from_storage, current_shard=shard.metadata
            ):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            read_items.append(
                ReadItem.create_for_tensor(
                    fqn=fqn,
                    storage_offsets=storage_offsets,
                    dest_offsets=dest_offsets,
                    lengths=lengths,
                    chunk=_chunk_for_sharmd(shard.metadata),
                )
            )
    return read_items


def create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) ->List[ReadItem]:
    if isinstance(md, BytesStorageMetadata):
        return [ReadItem.create_for_byteio(fqn, 0, 0, md.size_in_bytes)]

    elif isinstance(obj, ShardedTensor):
        local_shards = obj.local_shards()
    elif isinstance(obj, torch.Tensor):
        local_shards = [_create_shard_for(obj)]
    else:
        raise ValueError(
            f"Invalid checkpoint metadata for {fqn}, " +
            f"expected BytesStorageMetadata but found {type(md)}"
        )

    return _create_sharded_read_items(
        fqn,
        md,
        local_shards)

def create_default_read_plan(
    state_dict: Dict[str, Any],
    metadata: Metadata,
) -> LoadPlan:
    requests = []

    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensor
    """
    for fqn, obj in state_dict.items():
        md = metadata.state_dict_metadata[fqn]
        requests += create_read_items(metadata, fqn, md, obj)

    return LoadPlan(requests)

def create_default_global_read_plan(all_plans: List[LoadPlan]) -> List[LoadPlan]:
    return all_plans

def default_item_lookup(state_dict, fqn, chunk):
    obj = state_dict[fqn]
    if not isinstance(obj, ShardedTensor):
        return obj

    if chunk is None or chunk.offsets is None:
        raise ValueError(f"Cannot lookup {fqn} since its a ShardedTensor and no offset was provided")

    offsets = torch.Size(chunk.offsets)

    for shard in obj.local_shards():
        if torch.Size(shard.metadata.shard_offsets) == offsets:
            return shard.tensor

    raise ValueError(f"could not find shard at '{offsets}' for FQN: '{fqn}'")

def default_resolve_data(state_dict, fqn, chunk, is_bytesio):
    obj = default_item_lookup(state_dict, fqn, chunk)

    if is_bytesio:
        bytes = io.BytesIO()
        torch.save(obj, bytes)
        obj = bytes
    return obj

class DefaultSavePlanner(SavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        return create_default_local_plan(self.state_dict, self.is_coordinator)

    def create_global_plan(self, all_plans: List[SavePlan]) -> List[SavePlan]:
        self.global_plan, self.metadata = create_default_global_plan(all_plans)
        return self.global_plan

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        return new_plan

    def create_checkpoint_metadata(self, all_results: List[List[WriteResult]]) -> Metadata:
        populate_metadata_with_write_results(self.metadata, all_results)
        return self.metadata

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        return default_resolve_data(self.state_dict, write_item.fqn, write_item.chunk, write_item.is_bytesio)
 

class DefaultLoadPlanner(LoadPlanner):
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        return create_default_read_plan(self.state_dict, self.metadata)

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_read_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def write_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        self.state_dict[read_item.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = default_item_lookup(self.state_dict, read_item.fqn, read_item.chunk)
        tensor = tensor_narrow_n(tensor, read_item.dest_offsets, read_item.lengths)
        return tensor
