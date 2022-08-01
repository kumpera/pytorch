import copy
import bisect
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)


try:
    from spmd.tensor import DTensor as DistributedTensor, DeviceMesh
    from spmd.tensor.placement_types import Placement
    from spmd.tensor.placement_types import Shard as DTShard
    has_dt = True
except ImportError:
    has_dt = False

def _sharding_spec_to_offsets(
    sharding_spec: ShardingSpec, tensor_numel: int, world_size: int
) -> List[int]:
    r"""
    Translates the sharding spec to a list of offsets along dim 0. If the
    sharding spec is ChunkShardingSpec, only the ``dim`` is used and the
    placement is not used.
    """
    offsets: List[int] = []
    if isinstance(sharding_spec, EnumerableShardingSpec):
        for shard in sharding_spec.shards:
            offsets.append(shard.shard_offsets[0])
    elif isinstance(sharding_spec, ChunkShardingSpec):
        assert sharding_spec.dim == 0
        chunk_size = math.ceil(tensor_numel / world_size)
        if chunk_size == 1:
            offsets = [
                rank if rank < tensor_numel else tensor_numel
                for rank in range(world_size)
            ]
        else:
            offsets = [chunk_size if rank > 0 else 0 for rank in range(world_size)]
            offsets = list(itertools.accumulate(offsets))
    else:
        raise ValueError(f"Un-recognized sharding spec type {type(sharding_spec)}.")

    return offsets


def _offsets_to_split_sizes(
    input_offsets: List[int],
    output_offsets: List[int],
    tensor_numel: int,
    world_size: int,
    my_rank: int,
) -> Tuple[List[int], List[int]]:
    r"""
    Given the shard offsets for each rank of the input tensor and output tensor,
    this API returns the corresponding split sizes that can be passed to
    all_to_all_single().
    """

    def _get_interval(offsets):
        if my_rank != world_size - 1:
            return offsets[my_rank], offsets[my_rank + 1] - 1
        else:
            return offsets[my_rank], tensor_numel - 1

    def _offsets_to_sizes(offsets, begin, end):
        sizes = []
        for i, offset in enumerate(offsets):
            next_offset = offsets[i + 1] if i < len(offsets) - 1 else end + 1
            sizes.append(
                (next_offset - offset)
                - max(begin - offset, 0)
                - max(next_offset - end - 1, 0)
            )
        return sizes

    def _convert(from_offsets, to_offsets, split_sizes):
        begin, end = _get_interval(from_offsets)
        to_begin_rank = bisect.bisect(to_offsets, begin) - 1
        to_end_rank = bisect.bisect(to_offsets, end) - 1
        _split_sizes = _offsets_to_sizes(
            to_offsets[to_begin_rank : to_end_rank + 1], begin, end
        )
        split_sizes[to_begin_rank : to_end_rank + 1] = _split_sizes

    input_split_sizes = [0 for _ in range(world_size)]
    output_split_sizes = [0 for _ in range(world_size)]
    _convert(input_offsets, output_offsets, input_split_sizes)
    _convert(output_offsets, input_offsets, output_split_sizes)

    return input_split_sizes, output_split_sizes


def _reshard_flatten_tensor(
    input_tensor: ShardedTensor,
    output_spec: ShardingSpec,
    world_size: int,
    my_rank: int,
    device: torch.device,
    process_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """
    Resharded a sharded flatten tensor, this is used by FSDP to do sharded
    state_dict. But the functionaility is not supported by ShardedTensor.
    This API is designed to be used for FSDP; therefore this API supports only
    1-D ShardedTensor (hence the naming, reshard_flatten_tensor).

    This API uses the ChunkShardingSpec and EnumerableShardingSpec from
    torch.distributed.sharding_spec but ignores the placement field in
    ChunkShardingSpec, as the placement requires the callees understand the
    number of GPUs per node. The API simply uses the semantics of the sharding
    specs.

    Args:
        input_tensor (ShardedTensor): the original ShardedTensor. Must be 1D.
        output_spec (ShardingSpec): the sharding spect for the output tensor.
        world_size (int): total trainer count.
        my_rank (int): the rank for this trainer.

    Returns:
        The local shard for the new ShardedTensor.
    """

    input_spec = input_tensor.sharding_spec()
    size = input_tensor.size()
    if isinstance(size, int):
        raise ValueError("The input tensor has no dimensions.")
    tensor_numel = size.numel()
    input_offsets = _sharding_spec_to_offsets(input_spec, tensor_numel, world_size)
    output_offsets = _sharding_spec_to_offsets(output_spec, tensor_numel, world_size)
    input_split_sizes, output_split_sizes = _offsets_to_split_sizes(
        input_offsets, output_offsets, tensor_numel, world_size, my_rank
    )
    output_size = sum(output_split_sizes)
    local_shard = torch.empty(output_size, dtype=input_tensor.dtype, device=device)
    dist.all_to_all_single(
        local_shard,
        input_tensor.local_shards()[0].tensor,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=process_group,
    )
    return local_shard


def _all_gather_sharded_tensor(
    sharded_tensor: ShardedTensor, pg: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    if shards:
        local_tensor = shards[0].tensor.flatten()
        if not local_tensor.is_cuda:
            move_to_cpu = torch.ones(1, device=cuda_device)
            local_tensor = local_tensor.cuda()
        else:
            move_to_cpu = torch.zeros(1, device=cuda_device)
        num_padding = chunk_size - local_tensor.numel()
        if num_padding > 0:
            local_tensor = F.pad(local_tensor, [0, num_padding])
    else:
        local_tensor = torch.zeros(
            chunk_size, dtype=sharded_tensor.dtype, device=cuda_device
        )
        move_to_cpu = torch.zeros(1, device=cuda_device)

    tensor = torch.empty(
        chunk_size * world_size,
        dtype=local_tensor.dtype,
        device=cuda_device,
    )
    dist._all_gather_base(tensor, local_tensor, group=pg)

    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor


def _gather_state_dict(
    state_dict: Dict[str, Any],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors in the state_dict.
    """
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, ShardedTensor):
            output_tensor = _all_gather_sharded_tensor(tensor, pg)
            if tensor.local_shards() and tensor.local_shards()[0].tensor.is_cuda:
                tensor = output_tensor
            else:
                tensor = output_tensor.cpu()
        new_state_dict[key] = tensor
    return new_state_dict

def get_box(tensor: "DistributedTensor") -> Tuple:
    device_mesh = tensor.device_mesh
    assert device_mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"
    # assert tensor.placements[0].is_shard(), f"Only sharded placement supported"

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(dim=0)

    if tensor.placements[0].is_shard():
        shard_dim = placement.dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())

def get_box_for(tensor: "DistributedTensor", idx: int) -> Tuple:
    offsets, size = get_box(tensor)
    offsets = [val * idx for val in offsets]
    return (torch.Size(offsets), size)

def get_local_box(tensor: "DistributedTensor") -> Tuple:
    device_mesh = tensor.device_mesh
    return get_box_for(tensor, device_mesh.get_coordinate_on_dim(0))


def create_shard_md_from_dt(dt: "DistributedTensor", current_rank) -> ShardMetadata:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"

    offsets, sizes = get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}"
    )

def create_sharded_tensor_md_from_dt(dt: "DistributedTensor", dt_pg) -> ShardedTensorMetadata:
    # This is where it goes hilarious, we have to produce a ShardedTensor that has full coverage
    # and yet has only one valid shard for the current rank.

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1 #

    for i in range(shard_count):
        offsets, sizes = get_box_for(dt, i)
        shards_md.append(ShardMetadata(
            shard_offsets=list(offsets),
            shard_sizes=list(sizes),
            placement=f"rank:{scapegoat_rank}/{dt._local_tensor.device}"
        ))

    return ShardedTensorMetadata(
        shards_metadata=shards_md,
        size=dt.size(),
        tensor_properties=TensorProperties(
            dtype=dt.dtype,
            layout=dt.layout,
            requires_grad=dt.requires_grad,
            #ignore memory_format and pin_memory as those are not supported by DT
        )
    )

def get_dt_pg(dt: "DistributedTensor") -> dist.ProcessGroup:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"
    return mesh.get_dim_groups()[0]


def _create_chunk_sharded_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    device_per_node: int,
    pg: dist.ProcessGroup,
) -> ShardedTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local shard to create a ShardedTensor.

    This method handle sharding ShardedTensor and DistributedTensor
    """
    if isinstance(tensor, ShardedTensor):
        assert len(tensor.local_shards()) == 1

        inner_param = tensor.local_tensor()
        inner_st = _create_chunk_sharded_tensor_inner(
            inner_param,
            rank,
            world_size,
            torch.cuda.device_count(),
            pg,
        )

        outer_local_shard = tensor.local_shards()[0]
        shards: List[Shard] = [
            Shard(inner_st, copy.deepcopy(outer_local_shard.metadata))
        ]
        st_meta = copy.deepcopy(tensor.metadata())
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=tensor._process_group,
            init_rrefs=False
        )
        return st_outer
    elif has_dt and isinstance(tensor, DistributedTensor):
        device_mesh = tensor.device_mesh
        assert device_mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"

        inner_param = tensor._local_tensor

        inner_st = _create_chunk_sharded_tensor_inner(
            inner_param,
            rank,
            world_size,
            torch.cuda.device_count(),
            pg,
        )

        dt_pg = get_dt_pg(tensor)
        # We do this differently here, we create a ST with no local shards then patch it
        shards = []
        st_meta = create_sharded_tensor_md_from_dt(tensor, dt_pg)
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False
        )
        st_outer._local_shards.append(Shard(inner_st, create_shard_md_from_dt(tensor, dist.get_rank(dt_pg))))

        return st_outer

    else:
        return _create_chunk_sharded_tensor_inner(
            tensor,
            rank,
            world_size,
            device_per_node,
            pg)


def _create_chunk_sharded_tensor_inner(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    device_per_node: int,
    pg: dist.ProcessGroup,
) -> ShardedTensor:
    chunks = tensor.chunk(world_size, dim=0)
    if len(chunks) > rank:
        local_shard = chunks[rank].clone()
        offsets = [0 for _ in tensor.size()]
        offsets[0] = math.ceil(tensor.size()[0] / world_size) * rank
        local_shards = [Shard.from_tensor_and_offsets(local_shard, offsets, rank)]
    else:
        local_shards = []

    # Create a ShardedTensor without invoking communication.
    chunk_sizes = [list(chunk.size()) for chunk in chunks]
    dim0_offsets = [0] + list(
        itertools.accumulate([chunk_size[0] for chunk_size in chunk_sizes])
    )[:-1]
    offsets = [0] * (len(chunk_sizes[0]) - 1)
    chunk_offsets = [[d0] + offsets for d0 in dim0_offsets]
    placements = [
        f"rank:{r}/cuda:{r % device_per_node}" for r in range(len(chunk_sizes))
    ]
    assert len(chunk_sizes) == len(chunk_offsets) == len(placements)
    shard_metadata = [
        ShardMetadata(offset, size, placement)
        for offset, size, placement in zip(chunk_offsets, chunk_sizes, placements)
    ]

    # is_pinned = tensor.is_pinned() if isinstance(tensor, ShardedTensor) else False
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            # pin_memory=tensor.is_pinned(), # DT doesn't implement is_pinned
        )
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards,
        sharded_tensor_metadata=sharded_tensor_metadata,
        process_group=pg
    )
