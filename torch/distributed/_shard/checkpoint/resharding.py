import io
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
    TensorWriteData,
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

def _sharded_tensor_metadata(sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> TensorWriteData:
    return TensorWriteData(
        chunk=_chunk_for_sharmd(shard_md),
        info=_sharded_tensor_props_for(sharded_tensor)
    )

def _create_for_shardmd(fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> WriteItem:
    offsets = torch.Size(shard_md.shard_offsets)
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=_sharded_tensor_metadata(sharded_tensor, shard_md),
    )

def _create_for_shard(fqn: str, sharded_tensor: ShardedTensor, shard: Shard) -> WriteItem:
    offsets = torch.Size(shard.metadata.shard_offsets)
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=_sharded_tensor_metadata(sharded_tensor, shard.metadata),
    )

def _create_for_tensor(fqn: str, tensor: torch.Tensor) -> WriteItem:
    offsets = torch.Size([0] * len(tensor.size()))
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.TENSOR,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=offsets,
                sizes=tensor.size(),
                size_in_bytes=-1,
            ),
            info=TensorInfo(_tensor_props_for(tensor), tensor.size()),
        )
    )

def _create_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        index=MetadataIndex(fqn),
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
    The default plan creates a Metadata object with -1 as size_in_bytes.
    """
    md: Dict[str, STORAGE_TYPES] = dict()

    for plan in all_plans:
        for item in plan.items:
            if not item.is_shard:
                assert item.index.fqn not in md

            if item.is_bytesio:
                md[item.index.fqn] = BytesStorageMetadata(size_in_bytes=-1)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(item.index.fqn, TensorStorageMetadata(
                        properties=item.tensor_data.info,
                        chunks=[],
                    ))
                )

                item.index = MetadataIndex(
                    fqn=item.index.fqn,
                    offset=item.index.offset,
                    index=len(tensor_md.chunks))
                assert item.tensor_data.chunk is not None, f"Cannot create MD for tensor without bounds. FQN: {item.index.fqn}"
                tensor_md.chunks.append(item.tensor_data.chunk)

    return (all_plans, Metadata(md))

def find_chunk_index(list: List[ChunkStorageMetadata], index: MetadataIndex) -> int:
    # index fast path
    if index.index is not None:
        if len(list) > index.index and list[index.index] == index.offset:
            return index.index

    for i, c in enumerate(list):
        if c.offsets == index.offset:
            return i
    raise ValueError(f"Offset {index.offset} not found")

def find_shard(tensor: ShardedTensor, index: MetadataIndex) -> Shard:
    if index.offset is None:
        raise ValueError(f"Cannot lookup {index.fqn} since its a ShardedTensor and no offset was provided")

    shards = tensor.local_shards()
    # index fast path
    if index.index is not None:
        if len(shards) > index.index and torch.Size(shards[index.index].metadata.shard_offsets) == index.offset:
            return shards[index.index]

    for shard in shards:
        if torch.Size(shard.metadata.shard_offsets) == index.offset:
            return shard
    raise ValueError(f"could not find shard at '{index.offset}' for FQN: '{index.fqn}'")

def find_object(state_dict: Dict[str, Any], index: MetadataIndex) -> Any:
    obj = state_dict[index.fqn]
    if isinstance(obj, ShardedTensor):
        return find_shard(obj, index).tensor
    return obj


def populate_metadata_with_write_results(md: Metadata, results: List[List[WriteResult]]) -> None:
    """
    By default we populate the following:
        size_in_bytes of all leaf items
    """
    for wr_list in results:
        for wr in wr_list:
            item = md.state_dict_metadata[wr.index.fqn]
            if isinstance(item, TensorStorageMetadata):
                item.chunks[find_chunk_index(item.chunks, wr.index)].size_in_bytes = wr.size_in_bytes
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


def _create_sharded_read_items(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_shards: List[Shard],
) -> List[ReadItem]:

    read_items = []
    # this is a naive quadratic algo that can be optimized later
    for idx, shard in enumerate(local_shards):
        for storage_md in checkpoint_md.chunks:
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
                # FIXME pass the local shard index
                ReadItem.create_for_tensor(
                    index=MetadataIndex(fqn, torch.Size(shard.metadata.shard_offsets), idx),
                    storage_offsets=storage_offsets,
                    dest_offsets=dest_offsets,
                    lengths=lengths,
                )
            )
    return read_items


def create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    if isinstance(md, BytesStorageMetadata):
        return [ReadItem.create_for_byteio(
            index=MetadataIndex(fqn),
            src_offset=0,
            dest_offset=0,
            length=md.size_in_bytes
        )]

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
        requests += create_read_items(fqn, md, obj)

    return LoadPlan(requests)

def create_default_global_read_plan(all_plans: List[LoadPlan]) -> List[LoadPlan]:
    return all_plans


class DefaultSavePlanner(SavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        self.plan = create_default_local_plan(self.state_dict, self.is_coordinator)
        return self.plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> List[SavePlan]:
        self.global_plan, self.metadata = create_default_global_plan(all_plans)
        return self.global_plan

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def create_checkpoint_metadata(self, all_results: List[List[WriteResult]]) -> Metadata:
        populate_metadata_with_write_results(self.metadata, all_results)
        return self.metadata

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        if write_item.is_bytesio:
            bytes = io.BytesIO()
            torch.save(object, bytes)
            object = bytes
        return object


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
        self.state_dict[read_item.index.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.index)
        return self.transform_tensor(read_item, tensor)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return tensor_narrow_n(tensor, read_item.dest_offsets, read_item.lengths)
