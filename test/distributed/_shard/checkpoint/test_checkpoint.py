# Owner(s): ["oncall: distributed"]

from asyncore import write
import random
import sys
from typing import Optional, List, Union, Dict, Tuple, Any, cast
from torch.distributed._shard.checkpoint.storage import WriteResult

from torch.distributed._shard.checkpoint import (
    StorageReader,
    StorageWriter,
    CheckpointException,
    load_state_dict,
    save_state_dict,
)

import torch
import torch.distributed as dist
import torch.nn
import torch.futures
from torch.futures import Future
from torch.testing._internal.common_utils import TestCase

from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.checkpoint.state_dict_loader import (
    validate_metadata,
)

from torch.distributed._shard.checkpoint.state_dict_saver import (
    _create_metadata_from_local_state_dict,
)

from torch.distributed._shard.checkpoint.metadata import (
    Metadata,
    ShardedTensorStorageMetadata,
)

from torch.distributed._shard.checkpoint.storage import (
    LocalPlan,
    BytesReadRequest,
    BytesWriteRequest,
    TensorReadRequest,
    TensorWriteRequest,
)

from torch.distributed._shard.checkpoint.resharding import (
    default_prepare_writes
)

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

        metadata = _create_metadata_from_local_state_dict(module.state_dict())
        self.assertTrue(
            "regular" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        if dist.get_rank() == 0:
            print("------")
            print(metadata)
            print("------")
        module = TestModule()
        validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.extra_param = torch.nn.Parameter(torch.zeros(2, 2))
        with self.assertRaisesRegex(ValueError, "Could not find Tensor metadata"):
            validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.regular = torch.nn.Parameter(torch.zeros(2, 4))

        with self.assertRaisesRegex(ValueError, "Incompatible tensor size"):
            validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.extra_sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        with self.assertRaisesRegex(ValueError, "Could not find ShardedTensor metadata"):
            validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        with self.assertRaisesRegex(ValueError, "Incompatible ShardedTensor size"):
            validate_metadata(module.state_dict(), metadata)

    def gen_metadata(self) -> Metadata:
        module = TestModule()
        # compute the default saved metadata (must pass include_non_replicated_tensors or we'll get incomplete MD)
        metadata = _create_metadata_from_local_state_dict(module.state_dict())

        # _prepare only produc
        metadata = [metadata]
        dist.broadcast_object_list(metadata)

        return metadata[0]

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

        st_md = cast(ShardedTensorStorageMetadata, metadata.state_dict_metadata[".sharded"])
        sizes = st_md.shards[0].shard_metadata.shard_sizes
        for i in range(len(sizes)):
            sizes[i] = 1

        module = TestModule()
        with self.assertRaisesRegex(ValueError, "only has 1 available"):
            validate_metadata(module.state_dict(), metadata)

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

        st_md = cast(ShardedTensorStorageMetadata, metadata.state_dict_metadata[".sharded"])
        sizes = st_md.shards[0].shard_metadata.shard_sizes

        for i in range(len(sizes)):
            sizes[i] += 1

        module = TestModule()
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_metadata(module.state_dict(), metadata)


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_storage_type_mismatch(self) -> None:
        module = TestModule()

        metadata = self.gen_metadata()
        regular = metadata.state_dict_metadata["regular"]
        metadata.state_dict_metadata[".sharded"] = regular
        with self.assertRaisesRegex(ValueError, "ShardedTensorStorageMetadata but found"):
            validate_metadata(module.state_dict(), metadata)

        metadata = self.gen_metadata()
        sharded = metadata.state_dict_metadata[".sharded"]
        metadata.state_dict_metadata["regular"] = sharded
        with self.assertRaisesRegex(ValueError, "TensorStorageMetadata but found"):
            validate_metadata(module.state_dict(), metadata)


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_tensor_metadata_with_missing_rank_spec(self) -> None:
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:1/cuda:1",
            ],
        )

        st = sharded_tensor.zeros(spec, 4, 4, dtype=torch.float64)
        mapping = dict()

        md = _create_metadata_from_local_state_dict({ "st": st })

        st_md = md.state_dict_metadata["st"]
        self.assertEqual(1, len(st_md.shards))


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_storage_key_mapping(self) -> None:
        device = f"cuda:{dist.get_rank()}"
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        state_dict = {
            'sharded': sharded_tensor.rand(spec, (10, 10, )),
            'replicated': torch.rand(4, device=device),
            'bytes': [1, 2, 3, 4],
        }

        write_requests = _prepare(state_dict, write_replicated_data=self.rank == 0, request_prefix=f"{dist.get_rank()}_")

        if self.rank == 0:
            self.assertEqual(3, len(write_requests))
            self.assertTrue(any(r.request.fqn == "bytes" for r in write_requests))
            self.assertTrue(any(r.request.fqn == "replicated" for r in write_requests))
            self.assertTrue(any(r.request.fqn == "sharded" for r in write_requests))

        else:
            self.assertEqual(1, len(write_requests))
            self.assertTrue(any(r.request.fqn == "sharded" for r in write_requests))

class TestStorageBase:
    def __init__(
        self,
        fail_conf
    ):
        self.fail_conf = fail_conf
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def _get_ranks(self, name):
        return self.fail_conf[name] if name in self.fail_conf else None

    def _fail_rank(self, name):
        ranks = self._get_ranks(name)
        if ranks is not None and self.rank in ranks:
            raise ValueError(f"rank fail {self.rank} for {name}")

    def _fail_rank_async(self, name, requests=None):
        ranks = self._get_ranks(name)
        fut = Future()
        if ranks is not None and self.rank in ranks:
            fut.set_exception(ValueError(f"async rank fail {self.rank} for {name}"))
        else:
            results = [WriteResult(r.request_id,"") for r in requests] if requests is not None else None
            fut.set_result(results)
        return fut


class FaultyStorageWriter(TestStorageBase, StorageWriter):
    def __init__(
        self,
        fail_conf
    ):
        super(FaultyStorageWriter, self).__init__(fail_conf)


# fail_prepare_write
#fail_prepare_local_plan
#fail_prepare_global_plan

    def prepare(self) -> None:
        self._fail_rank("fail_prepare")

    def prepare_local_plan(self, plan: LocalPlan) -> LocalPlan:
        self._fail_rank("fail_prepare_local_plan")
        return plan

    def prepare_global_plan(self, plans: List[LocalPlan]) -> List[LocalPlan]:
        self._fail_rank("fail_prepare_global_plan")
        return plans

    def prepare_writes(
        self,
        state_dict: Dict[str, Any],
        plan: LocalPlan,
    ) -> Tuple[List[TensorWriteRequest], List[BytesWriteRequest]]:
        self._fail_rank("fail_prepare_write")
        return default_prepare_writes(state_dict, plan)

    def write_data(
        self,
        storage_plan: _StoragePrefix,
        tensors: List[TensorWriteRequest],
        bytes: List[BytesWriteRequest]
    ) -> Future[List[WriteResult]]:
        self._fail_rank("fail_write_data_on_ranks")
        return self._fail_rank_async("fail_write_data_on_ranks_async", tensors + bytes)

    # def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
    #     self._fail_rank("fail_write_bytes_on_ranks")
    #     return self._fail_rank_async("fail_write_bytes_on_ranks_async", requests)

    # def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
    #     self._fail_rank("fail_write_tensors_on_ranks")
    #     return self._fail_rank_async("fail_write_tensors_on_ranks_async", requests)

    def finish(self, metadata: Metadata) -> None:
        self._fail_rank("fail_finish")

    def prepare_storage(self, storage_writes: List[Union[TensorWriteRequest, BytesWriteRequest]]) -> None:
        self._fail_rank("fail_prepare_storage")

class FaultyStorageReader(TestStorageBase, StorageReader):
    def __init__(
        self,
        metadata,
        fail_conf
    ):
        super(FaultyStorageReader, self).__init__(fail_conf)
        self.metadata = metadata

    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        self._fail_rank("fail_read_bytes")
        bad_ranks = self._get_ranks("fail_deser_bytes")
        for r in requests:
            if bad_ranks is not None and self.rank in bad_ranks:
                # this is not "guaranteed" to fail, but hard to beat
                rand = random.Random(1237)
                r.bytes.write(rand.randbytes(32))
            else:
                torch.save([1, 2, 3], r.bytes)

        return self._fail_rank_async("fail_read_bytes_async")

    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        self._fail_rank("fail_read_tensors")
        return self._fail_rank_async("fail_read_tensors_async")

    def read_metadata(self) -> Metadata:
        self._fail_rank("fail_read_metadata")
        return self.metadata

class TestDistributedFailure(ShardedTensorTestBase):
    def get_spec(self):
        return ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{r}/cuda:{r}" for r in range(dist.get_world_size())
            ]
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_dummy_writer_works(self) -> None:
        state_dict = {
            'sharded': sharded_tensor.rand(self.get_spec(), 20, 20),
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        save_state_dict(state_dict, FaultyStorageWriter({}))


    def _test_dist_failure(self, callback, kwargs):
        bad_ranks = list(kwargs.values())[0] if len(kwargs) > 0 else []

        # Empty bad_ranks means it must work
        if len(bad_ranks) == 0:
            callback()
        else:
            with self.assertRaises(CheckpointException) as cm:
                callback()
            e = cm.exception
            for rank, ex in e.failures.items():
                self.assertTrue(rank in bad_ranks, msg=f"{rank} did not fail")
                if not kwargs.get("ignore_exception_type", False):
                    self.assertEqual(ValueError, type(ex), str(ex))

            failed_ranks = e.failures.keys()
            for rank in bad_ranks:
                self.assertTrue(rank in failed_ranks, msg=f"{rank} was supposed to fail was fine")


    def _test_save(self, state_dict, coordinator=0, **kwargs):
        no_dist = not dist.is_initialized()

        def _save():
            save_state_dict(
                state_dict,
                storage_writer=FaultyStorageWriter(kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )
        self._test_dist_failure(_save, kwargs)

    def _test_load(self, state_dict, coordinator=0, **kwargs):
        no_dist = not dist.is_initialized()

        def _load():
            metadata = _create_metadata_from_local_state_dict(state_dict)
            load_state_dict(
                state_dict,
                storage_reader=FaultyStorageReader(metadata, kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        self._test_dist_failure(_load, kwargs)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_save_error_handling(self) -> None:
        state_dict = {
            'sharded': sharded_tensor.rand(self.get_spec(), 20, 20),
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        self._test_save(state_dict, fail_prepare=[0])
        self._test_save(state_dict, fail_finish=[0])

        self._test_save(state_dict, fail_prepare_storage=[0])
        self._test_save(state_dict, fail_write_tensors_on_ranks=[1])
        self._test_save(state_dict, fail_write_tensors_on_ranks_async=[2])
        self._test_save(state_dict, fail_write_bytes_on_ranks=[3])
        self._test_save(state_dict, fail_write_bytes_on_ranks_async=[1])

        self._test_save(state_dict, fail_write_tensors_on_ranks_async=[1, 3])

        self._test_save(state_dict, coordinator=1, fail_prepare=[1])
        self._test_save(state_dict, coordinator=1, fail_finish=[1])


    def test_save_error_handling_no_dist(self) -> None:
        state_dict = {
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        self.assertFalse(dist.is_initialized())

        self._test_save(state_dict, fail_prepare=[0])
        self._test_save(state_dict, fail_finish=[0])

        self._test_save(state_dict, fail_prepare_storage=[0])
        self._test_save(state_dict, fail_write_tensors_on_ranks=[0])
        self._test_save(state_dict, fail_write_tensors_on_ranks_async=[0])
        self._test_save(state_dict, fail_write_bytes_on_ranks=[0])
        self._test_save(state_dict, fail_write_bytes_on_ranks_async=[0])

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_load_error_handling(self) -> None:
        state_dict = {
            'sharded': sharded_tensor.rand(self.get_spec(), 20, 20),
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        self._test_load(state_dict)
        self._test_load(state_dict, fail_read_metadata=[0])
        self._test_load(state_dict, fail_read_bytes=[1])
        self._test_load(state_dict, fail_read_bytes_async=[2])
        self._test_load(state_dict, fail_read_tensors=[3])
        self._test_load(state_dict, fail_read_tensors_async=[1])
        # We don't want to depend on the actual exception raised by pickle
        self._test_load(state_dict, fail_deser_bytes=[2], ignore_exception_type=True)

        self._test_load(state_dict, coordinator=1, fail_read_metadata=[3])
        self._test_load(state_dict, coordinator=2, fail_read_bytes=[0])
        self._test_load(state_dict, coordinator=3, fail_read_tensors_async=[2])


    def test_load_error_handling_no_dist(self) -> None:
        state_dict = {
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }
        self._test_load(state_dict)
        self._test_load(state_dict, fail_read_metadata=[0])
        self._test_load(state_dict, fail_read_bytes=[0])
        self._test_load(state_dict, fail_read_bytes_async=[0])
        self._test_load(state_dict, fail_read_tensors=[0])
        self._test_load(state_dict, fail_read_tensors_async=[0])
        self._test_load(state_dict, fail_deser_bytes=[0], ignore_exception_type=True)
if __name__ == "__main__":
    run_tests()
