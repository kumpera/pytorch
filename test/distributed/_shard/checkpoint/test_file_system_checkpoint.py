# Owner(s): ["oncall: distributed"]

import sys
import shutil
import tempfile
from typing import Dict, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.checkpoint.api import CheckpointException
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    MyShardedModel1
)

from contextlib import contextmanager


from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed._shard.checkpoint.filesystem import (
    _StorageInfo,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

def _sharded_tensor_gather(
        self,
        dst=0,
        out=None,
):
    return self.gather(dst=dst, out=out)

def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: Dict[str, torch.Tensor],
    state_dict_2: Dict[str, torch.Tensor],
) -> bool:
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2), f"Key {key}'s tensor does not match"
            )
        elif isinstance(value_1, ShardedTensor):
            for local_shard_1, local_shard_2 in zip(
                value_1.local_shards(), value_2.local_shards()
            ):
                self.assertTrue(
                    torch.equal(local_shard_1.tensor, local_shard_1.tensor),
                    f"Key {key}'s shard does not match",
                )

    return True


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# The ShardedModels are borrowed from test/distributed/_sharded_tensor/test_sharded_tensor.py
class MyShardedModel3(torch.nn.Module):
    def __init__(
        self,
        spec: ShardingSpec,
    ) -> None:
        super(MyShardedModel3, self).__init__()
        self.sharded_tensor: ShardedTensor = sharded_tensor.rand(
            spec, 10, 20, init_rrefs=False
        )


class TestSaveLoadNoDist(TestCase):
    def test_read_write_only_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path)
            save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer, no_dist=True)

            state_dict_to_load_to = MyTestModule().state_dict()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader, no_dist=True)

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

    def test_save_fails_tensor_file_already_exists(self):
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = {
                "tensor": torch.rand(10)
            }

            fs_writer = FileSystemWriter(path=path)
            md = save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer, no_dist=True)
            storage_md = cast(_StorageInfo, list(md.storage_data.values())[0])

            with self.assertRaisesRegex(CheckpointException, "write") as cm:
                fs_writer = FileSystemWriter(path=path)
                save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer, no_dist=True)

            exc: CheckpointException = cm.exception
            self.assertIn(0, exc.failures)
            nested = exc.failures[0]
            self.assertIsInstance(nested, FileExistsError)
            self.assertIn(storage_md.relative_path, str(nested))

class TestDistributedStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_read_write_shard_tensor(self) -> None:
        paths = [tempfile.mkdtemp()]
        dist.broadcast_object_list(paths)

        path = paths[0]

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec, init_rrefs=False)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()


@contextmanager
def dist_tempdir():
    rank = dist.get_rank()
    paths = [tempfile.mkdtemp()] if rank == 0 else [None]
    dist.broadcast_object_list(paths)

    try:
        yield paths[0]
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(paths[0], ignore_errors=True)

class TestDistributedReshardOnLoad(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def load_tensor(self, tensor: ShardedTensor) -> torch.Tensor:
        res = torch.zeros(tensor.shape, device="cuda:0") if dist.get_rank() == 0 else None
        _sharded_tensor_gather(tensor, out=res)
        return cast(Tensor, res)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_load_with_different_shard_plan(self) -> None:
        # We hardcode the assumption of how many shards are around
        self.assertEqual(self.world_size, dist.get_world_size())

        specs = [
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                ],
            ),
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:1/cuda:1",
                    "rank:0/cuda:0",
                ],
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0/cuda:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[2, 0],
                        shard_sizes=[1, 20],
                        placement="rank:1/cuda:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[3, 0],
                        shard_sizes=[3, 20],
                        placement="rank:0/cuda:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[6, 0],
                        shard_sizes=[3, 20],
                        placement="rank:1/cuda:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[9, 0],
                        shard_sizes=[1, 20],
                        placement="rank:0/cuda:0",
                    ),
                ]
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[8, 20],
                        placement="rank:1/cuda:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0/cuda:0",
                    ),
                ]
            ),
        ]

        for s0 in specs:
            for s1 in specs:
                if s0 == s1:
                    continue

                with dist_tempdir() as path:
                    model_to_save = MyShardedModel3(s0)
                    model_to_save._register_state_dict_hook(state_dict_hook)
                    state_dict_to_save = model_to_save.state_dict()

                    fs_writer = FileSystemWriter(path=path)
                    save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

                    dist.barrier()

                    model_to_load = MyShardedModel3(s1)
                    model_to_load._register_state_dict_hook(state_dict_hook)
                    state_dict_to_load_to = model_to_load.state_dict()
                    dist.barrier()

                    fs_reader = FileSystemReader(path=path)
                    load_state_dict(
                        state_dict=state_dict_to_load_to, storage_reader=fs_reader
                    )

                    dist.barrier()
                    store_tensor = self.load_tensor(model_to_save.sharded_tensor)
                    dist.barrier()
                    load_tensor = self.load_tensor(model_to_load.sharded_tensor)

                    if dist.get_rank() == 0:
                        self.assertTrue(
                            torch.allclose(store_tensor, load_tensor), msg=f"{s0} vs {s1}"
                        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_load_rowwise_to_colwise(self) -> None:
        self.assertEqual(self.world_size, dist.get_world_size())

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        src_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        dst_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        with dist_tempdir() as path:
            model_to_save = MyShardedModel3(src_spec).cuda(dist.get_rank())
            model_to_save._register_state_dict_hook(state_dict_hook)
            state_dict_to_save = model_to_save.state_dict()

            fs_writer = FileSystemWriter(path=path)
            save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

            model_to_load = MyShardedModel3(dst_spec).cuda(dist.get_rank())
            model_to_load._register_state_dict_hook(state_dict_hook)
            state_dict_to_load_to = model_to_load.state_dict()

            fs_reader = FileSystemReader(path=path)

            load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

            # We can't use torch.allclose since each ST has a different sharding spec
            store_tensor = self.load_tensor(model_to_save.sharded_tensor)
            load_tensor = self.load_tensor(model_to_load.sharded_tensor)

            if dist.get_rank() == 0:
                self.assertTrue(torch.allclose(store_tensor, load_tensor))



    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_save_load_bytes(self) -> None:
        state_dict_to_save = {
            'bytes0': [1],
            'bytes1': 'string'
        }

        with dist_tempdir() as path:
            fs_writer = FileSystemWriter(path=path)
            save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

            state_dict_to_load = {
                'bytes0': [2],
                'bytes1': 'other'
            }

            fs_reader = FileSystemReader(path=path)
            load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

            self.assertEqual([1], state_dict_to_load['bytes0'])
            self.assertEqual('string', state_dict_to_load['bytes1'])


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_switch_between_sharded_tensor_to_tensor(self) -> None:
        tensor_size = 32

        specs = [
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                ],
            ),
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:1/cuda:1",
                    "rank:0/cuda:0",
                ],
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[8],
                        placement="rank:1/cuda:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8],
                        shard_sizes=[tensor_size - 8],
                        placement="rank:0/cuda:0",
                    ),
                ]
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[10],
                        placement="rank:0/cuda:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[10],
                        shard_sizes=[tensor_size - 10],
                        placement="rank:1/cuda:1",
                    ),
                ]
            ),
        ]

        for save_spec in specs:
            for load_spec in specs:
                print(f"testing from {save_spec} to {load_spec}")
                save_dict = {
                    'sharded': sharded_tensor.rand(save_spec, tensor_size),
                    'replicated': torch.rand(tensor_size, device=self.rank)
                }

                with dist_tempdir() as path:
                    fs_writer = FileSystemWriter(path=path)
                    md = save_state_dict(state_dict=save_dict, storage_writer=fs_writer)

                    # Freaky Friday the tensors
                    load_dict = {
                        'sharded': torch.zeros(tensor_size, device=self.rank),
                        'replicated': sharded_tensor.zeros(load_spec, tensor_size)
                    }

                    fs_reader = FileSystemReader(path=path)
                    load_state_dict(state_dict=load_dict, storage_reader=fs_reader)

                    save_dict_sharded = self.load_tensor(save_dict['sharded'])
                    load_dict_replicated = self.load_tensor(load_dict['replicated'])

                    if dist.get_rank() == 0:
                        self.assertTrue(
                            torch.allclose(save_dict_sharded, load_dict['sharded']),
                            f"save-spec {save_spec} load-spec {load_spec}"
                        )
                        self.assertTrue(
                            torch.allclose(save_dict['replicated'], load_dict_replicated),
                            f"save-spec {save_spec} load-spec {load_spec}"
                        )

if __name__ == "__main__":
    run_tests()
