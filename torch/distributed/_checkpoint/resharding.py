from typing import List, Tuple

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import (
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)

from .metadata import (
    TensorReadRequest,
    ShardStorageMetadata,
    ShardedTensorStorageMetadata,
    TensorWriteRequest,
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


def _compute_sharded_tensor_md(
    storage_key_prefix: str, tensor: ShardedTensor
) -> ShardedTensorStorageMetadata:
    smd = []
    for shard_md in tensor.metadata().shards_metadata:
        # each shard is in it own storage key.
        # Most network file system is optimized with single write, multiple read
        # Unless we can group tensors locally into one big chunk
        # It might be best to write each shard as one key
        suffix = "_".join([str(i) for i in shard_md.shard_offsets])
        storage_key = f"{storage_key_prefix}_{suffix}"

        shard_size = 1
        for d in shard_md.shard_sizes:
            shard_size *= d

        # not particularly great
        storage_size = shard_size * tensor.local_shards()[0].tensor.element_size()

        one_smd = ShardStorageMetadata(
            shard_metadata=shard_md,
            storage_key=storage_key,
            length=storage_size,
        )
        smd.append(one_smd)

    return ShardedTensorStorageMetadata(
        tensor_metadata=tensor.metadata(),
        storage_metadata=smd,
    )


def _prepare_sharded_tensor_write(
    sharded_tensor: ShardedTensor,
    storage_key_prefix: str,
) -> Tuple[List[TensorWriteRequest], ShardedTensorStorageMetadata]:
    """
    Prepare sharded tensor write.

    Args:
        sharded_tensor: The sharded tensor to persist.
        storage_key_prefix: The identifier to be prepended to storage keys
                            associated with the sharded tensor.

    Returns:
        Write requests for persisting the sharded tensor, and metadata
        describing the persisted sharded tensor.
    """
    write_requests = []
    for shard in sharded_tensor.local_shards():
        # each shard has its own storage key.
        # For most cases, the read is a recovery from a failure to the same sharding
        # and does not need any resharding, write each shard as is is the most effective
        suffix = "_".join([str(i) for i in shard.metadata.shard_offsets])
        storage_key = f"{storage_key_prefix}_{suffix}"

        tensor = shard.tensor.detach()

        wr = TensorWriteRequest(
            tensor=tensor,
            storage_key=storage_key,
        )
        write_requests.append(wr)
    return write_requests, _compute_sharded_tensor_md(
        storage_key_prefix, sharded_tensor
    )


def _prepare_sharded_tensor_read(
    metadata: ShardedTensorStorageMetadata, sharded_tensor_out: ShardedTensor
) -> List[TensorReadRequest]:
    """
    Prepare sharded tensor read.

    Args:
        metadata: Metadata describing the persisted sharded tensor. Normally,
                  this is generated by func::`_prepare_sharded_tensor_write`.
        sharded_tensor_out: The dest sharded tensor.

    Returns:
        A list of class::`TensorReadRequest`. When fullfilled,
        `sharded_tensor_out`'s local shards load from the persisted sharded
        tensor.
    """
    read_reqs = []
    for shard in sharded_tensor_out.local_shards():
        # scan all mds looking for chunks
        for storage_md in metadata.storage_metadata:
            shard_md_from_storage = storage_md.shard_metadata
            tensor = shard.tensor.detach()
            assert shard_md_from_storage is not None
            # this is a naive quadratic algo that can later be optimized

            # do they overlap?
            if not _check_shard_metadata_pair_overlap(
                shard.metadata, shard_md_from_storage
            ):
                continue

            storage_key = storage_md.storage_key

            target_tensor = tensor
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
                    tensor=target_tensor,
                    storage_key=storage_key,
                    offsets=tuple(offsets),
                    lengths=tuple(lengths),
                )
            )
    return read_reqs
