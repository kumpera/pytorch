# Owner(s): ["oncall: distributed"]

import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn

from torch.distributed._shard.checkpoint.state_dict_loader import validate_metadata
from torch.distributed._shard.checkpoint.state_dict_saver import _prepare
from torch.distributed._shard.checkpoint.metadata import Metadata

from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import (
    state_dict_hook,
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)

from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)



class TestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sharded: ShardedTensor = sharded_tensor.zeros(self.spec(), 4, 4)
        self.regular = torch.nn.Parameter(torch.ones(4, 4))
        self.extra_sharded: Optional[ShardedTensor] = None
        self.extra_param: Optional[torch.nn.Parameter] = None
        self._register_state_dict_hook(state_dict_hook)

    def spec(self) -> ChunkShardingSpec:
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        return ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )


class TestDistributedCheckpointing(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_validate_metadata(self) -> None:
        module = TestModule()

        # compute the default saved metadata (must pass include_non_replicated_tensors or we'll get incomplete MD)
        metadata, _, _, _ = _prepare(module.state_dict(), include_non_replicated_tensors=True)
        self.assertTrue(
            "regular" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        module = TestModule()
        self.assertIsNone(validate_metadata(module.state_dict(), metadata))

        module = TestModule()
        module.extra_param = torch.nn.Parameter(torch.zeros(2, 2))
        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Could not find Tensor metadata" in res[0])
        self.assertTrue("extra_param" in res[0])

        module = TestModule()
        module.regular = torch.nn.Parameter(torch.zeros(2, 4))

        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Incompatible tensor size" in res[0])
        self.assertTrue("regular" in res[0])

        module = TestModule()
        module.extra_sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Could not find ShardedTensor metadata" in res[0])
        self.assertTrue("extra_sharded" in res[0])

        module = TestModule()
        module.sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        res = validate_metadata(module.state_dict(), metadata)
        self.assertIsNotNone(res)
        self.assertTrue("Incompatible ShardedTensor size" in res[0])
        self.assertTrue("sharded" in res[0])

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_metadata_is_different_across_ranks(self) -> None:
        module = TestModule()
        # compute the default saved metadata (must pass include_non_replicated_tensors or we'll get incomplete MD)
        metadata, _, _, _ = _prepare(module.state_dict(), include_non_replicated_tensors=False)

        # _prepare skips tensors when rank > 0
        if dist.get_rank() == 0:
            self.assertTrue(
                "regular" in metadata.state_dict_metadata,
                f"keys: {metadata.state_dict_metadata.keys()}",
            )
        else:
            self.assertTrue(
                "regular" not in metadata.state_dict_metadata,
                f"keys: {metadata.state_dict_metadata.keys()}",
            )

    def gen_metadata(self) -> Metadata:
        module = TestModule()
        # compute the default saved metadata (must pass include_non_replicated_tensors or we'll get incomplete MD)
        metadata, _, _, _ = _prepare(module.state_dict(), include_non_replicated_tensors=True)
        return metadata

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_shard_too_small(self) -> None:
        metadata = self.gen_metadata()

        # we make the first stored shard smaller
        self.assertTrue(
            ".sharded" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        sizes = (
            metadata.state_dict_metadata[".sharded"]
            .storage_metadata[0]
            .shard_metadata.shard_sizes
        )
        for i in range(len(sizes)):
            sizes[i] = 1

        module = TestModule()
        self.assertIsNotNone(validate_metadata(module.state_dict(), metadata))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_shard_overlap(self) -> None:
        metadata = self.gen_metadata()

        # we make the first stored shard smaller
        self.assertTrue(
            ".sharded" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        sizes = (
            metadata.state_dict_metadata[".sharded"]
            .storage_metadata[0]
            .shard_metadata.shard_sizes
        )
        for i in range(len(sizes)):
            sizes[i] += 1

        module = TestModule()
        self.assertIsNotNone(validate_metadata(module.state_dict(), metadata))



    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_storage_type_mismatch(self) -> None:
        module = TestModule()

        metadata = self.gen_metadata()
        regular = metadata.state_dict_metadata["regular"]
        metadata.state_dict_metadata[".sharded"] = regular
        res = validate_metadata(module.state_dict(), metadata)

        self.assertIsNotNone(res)
        self.assertTrue("Expected ShardedTensorStorageMetadata but found" in res[0], res[0])

        metadata = self.gen_metadata()
        sharded = metadata.state_dict_metadata[".sharded"]
        metadata.state_dict_metadata["regular"] = sharded
        res = validate_metadata(module.state_dict(), metadata)
        self.assertTrue("Expected TensorStorageMetadata but found" in res[0], res[0])

        self.assertIsNotNone(res)

if __name__ == "__main__":
    run_tests()
