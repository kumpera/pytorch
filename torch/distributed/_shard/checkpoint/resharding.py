import io
from typing import List, Tuple, Dict, Any, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist

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
    LocalPlan,
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

def _create_for_shardmd(fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> WriteItem:
    return WriteItem(
        fqn=fqn,
        type=WriteItemType.SHARD,
        chunk=ChunkStorageMetadata(
            offsets=torch.Size(shard_md.shard_offsets),
            sizes=torch.Size(shard_md.shard_sizes),
            size_in_bytes=-1,
        ),
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
    return LocalPlan(requests)

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
                requests.append(_create_for_bytesio(fqn, obj))
    return LocalPlan(requests)

def create_default_global_plan(all_plans: List[LocalPlan]) -> Tuple[List[LocalPlan], Metadata]:
    """
    The default plan creates a Metadata object with -1 as size_in_bytes
    It modifies WriteItem::chunk_index.
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

                idx = len(tensor_md.chunks)
                assert item.chunk is not None, f"Cannot create MD for tensor without bounds. FQN: {item.fqn}"
                tensor_md.chunks.append(item.chunk)
                item.chunk_index = idx

    return (all_plans, Metadata(md))

def populate_metadata_with_write_results(md: Metadata, results: List[List[WriteResult]]) -> None:
    """
    By default we populate the following:
        size_in_bytes of all leaf items
        md::planner_data with a Dict[MetadataIndex, Any] by aggregating over WriteResults
        md::storage_data with a Dict[MetadataIndex, Any] by aggregating over WriteResults
    """
    planner_data = dict()
    storage_data = dict()

    for wr_list in results:
        for wr in wr_list:
            item = md.state_dict_metadata[wr.fqn]
            if isinstance(item, TensorStorageMetadata):
                assert wr.chunk_index is not None
                item.chunks[wr.chunk_index].size_in_bytes = wr.size_in_bytes
            else:
                item.size_in_bytes = wr.size_in_bytes
            if wr.planner_data is not None:
                planner_data[MetadataIndex(wr.fqn, wr.chunk_index)] = wr.planner_data
            if wr.storage_data is not None:
                storage_data[MetadataIndex(wr.fqn, wr.chunk_index)] = wr.storage_data

    if len(planner_data) > 0:
        md.planner_data = planner_data
    if len(storage_data) > 0:
        md.storage_data = storage_data

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
    if md.planner_data is not None:
        planner_data = md.planner_data.get(key, None)
    storage_data = None
    if md.storage_data is not None:
        storage_data = md.storage_data.get(key, None)
    return (planner_data, storage_data)


def _create_read_items(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_shards: List[Shard],
    metadata: Metadata
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

            item_idx = MetadataIndex(fqn, idx)
            planner_data, storage_data = _get_extra_metadata(metadata, item_idx)

            read_items.append(
                ReadItem(
                    fqn=fqn,
                    chunk=storage_md,
                    chunk_index=idx,
                    storage_offsets=torch.Size(storage_offsets),
                    dest_offsets=torch.Size(dest_offsets),
                    lengths=torch.Size(lengths),
                    planner_data=planner_data,
                    storage_data=storage_data,
                )
            )
    return read_items

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


        if isinstance(md, BytesStorageMetadata):
            item_idx = MetadataIndex(fqn, None)
            planner_data, storage_data = _get_extra_metadata(metadata, item_idx)
            requests.append(ReadItem(
                fqn=fqn,
                storage_offsets=torch.Size([0]),
                dest_offsets=torch.Size([0]),
                lengths=torch.Size([md.size_in_bytes]),
                planner_data=planner_data,
                storage_data=storage_data,
            ))
            continue
        elif isinstance(obj, ShardedTensor):
            local_shards = obj.local_shards()
        elif isinstance(obj, torch.Tensor):
            tensor = obj.detach()
            local_shards = [_create_shard_for(tensor)]
        else:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, " +
                f"expected BytesStorageMetadata but found {type(md)}"
            )

        requests += _create_read_items(
            fqn,
            md,
            local_shards,
            metadata)

    return LoadPlan(requests)

def create_default_global_read_plan(all_plans: List[LoadPlan]) -> List[LoadPlan]:
    return all_plans

def default_item_lookup(state_dict, item):
    obj = state_dict[item.fqn]
    if not isinstance(obj, ShardedTensor):
        return obj

    if item.chunk is None or item.chunk.offsets is None:
        raise ValueError(f"Cannot lookup {item.fqn} since its a ShardedTensor and no offset was provided")

    offsets = torch.Size(item.chunk.offsets)

    for shard in obj.local_shards():
        if torch.Size(shard.metadata.shard_offsets) == offsets:
            return shard.tensor
    raise ValueError(f"could not find shard at '{offsets}' for FQN: '{item.fqn}'")

class DefaultSavePlanner(SavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LocalPlan:
        return create_default_local_plan(self.state_dict, self.is_coordinator)

    def create_global_plan(self, all_plans: List[LocalPlan]) -> List[LocalPlan]:
        self.global_plan, self.metadata = create_default_global_plan(all_plans)
        return self.global_plan

    def finish_plan(self, new_plan: LocalPlan) -> LocalPlan:
        return new_plan

    def create_checkpoint_metadata(self, all_results: List[List[WriteResult]]) -> Metadata:
        populate_metadata_with_write_results(self.metadata, all_results)
        return self.metadata

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        obj = default_item_lookup(self.state_dict, write_item)

        # obj = write_item.lookup(self.state_dict)
        if write_item.is_bytesio:
            bytes = io.BytesIO()
            torch.save(obj, bytes)
            obj = bytes
        return obj


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
        tensor = default_item_lookup(self.state_dict, read_item)
        tensor = tensor_narrow_n(tensor, read_item.dest_offsets, read_item.lengths)
        return tensor
