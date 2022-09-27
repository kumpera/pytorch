# Owner(s): ["oncall: distributed"]
import copy
import sys
from typing import Any, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed._shard.sharding_spec as shard_spec
from torch import distributed as dist
from torch.distributed.remote_device import _remote_device
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_optim import (
    named_params_with_sharded_tensor,
    ShardedOptimizer,
)
from torch.distributed._shard.sharded_tensor.api import (
    Shard,
    ShardedTensor,
)
from torch.distributed._shard.sharding_plan import ShardingPlan
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp._tensor_flattener import (
    _set_tensor_flattener,
    TensorFlattener,
)
from torch.distributed.fsdp._utils import _set_fsdp_flattened
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

LR = 3e-5
OPS_NOT_SHARD = [
    "net3.weight",
    "net3.bias",
]
SHARD_PARAMS = [
    "net1.weight",
    "net2.weight",
]


class STShardingInfo(NamedTuple):
    """:class:`ShardedTensor` sharding information."""

    sharding_spec: shard_spec.ShardingSpec
    global_size: torch.Size
    process_group: dist.ProcessGroup


class ShardedTensorFlattener(TensorFlattener):
    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if type(tensor) is ShardedTensor:
            sharding_info = STShardingInfo(
                tensor.sharding_spec(), tensor.size(), tensor._process_group
            )
            local_tensor = tensor.local_tensor()
            return local_tensor, sharding_info
        return tensor, None

    def post_unflatten_transform(
        self, tensor: torch.Tensor, sharding_info: STShardingInfo
    ) -> torch.Tensor:
        sharded_tensor = ShardedTensor._init_from_local_tensor(
            tensor,
            _rewrite_spec_if_needed(
                sharding_info.sharding_spec,
                tensor,
                dist.get_rank(sharding_info.process_group)
            ),
            sharding_info.global_size,
            process_group=sharding_info.process_group,
        )
        _set_fsdp_flattened(sharded_tensor)
        return sharded_tensor


    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
    ) -> torch.Tensor:
        if type(tensor) is ShardedTensor:
            assert len(tensor.local_shards()) == 1

            inner_param = tensor.local_tensor()
            inner_st = _create_chunk_sharded_tensor(
                inner_param,
                rank,
                world_size,
                num_devices_per_node,
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
        else:
            return _create_chunk_sharded_tensor(
                tensor,
                rank,
                world_size,
                num_devices_per_node,
                pg
            )

    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        shards = tensor.local_shards()
        # default impl removes this line
        if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
            tensor = shards[0].tensor
            shards = tensor.local_shards()

        return (tensor, [shards[0].tensor] if len(shards) > 0 else [])

_set_tensor_flattener(ShardedTensorFlattener())

def _rewrite_spec_if_needed(spec: shard_spec.ShardingSpec, tensor: torch.Tensor, rank: int):
    """
    Rewrite ``spec`` to match the device of ``tensor``.

    FSDP.sharded_optim_state_dict sneakly ships optimizer state to CPU so if the original ShardingSpec
    produces CUDA metadata, ST construction bombs.
    """
    if not isinstance(spec, ChunkShardingSpec):
        return spec

    # let's see if we need
    rewrite = False
    for p in spec.placements:
        if p.rank() == rank and p.device() != tensor.device:
            rewrite = True
            break
    if rewrite:
        spec = copy.deepcopy(spec)
        for i, placement in enumerate(spec.placements):
            if placement.rank() == rank and placement.device() != tensor.device:
                spec.placements[i] = _remote_device(f"rank:{rank}/{tensor.device}")

    return spec

def _generate_chunk_sharding_spec(world_size, tp_pg):
    placements = [
        f"rank:{idx}/cuda:{dist.distributed_c10d.get_global_rank(tp_pg, idx) % torch.cuda.device_count()}"
        for idx in range(world_size)
    ]
    colwise_spec = ChunkShardingSpec(
        dim=0,
        placements=placements,
    )
    rowwise_spec = ChunkShardingSpec(
        dim=1,
        placements=placements,
    )
    return colwise_spec, rowwise_spec

def _is_nested_tensor(val: Any) -> bool:
    if type(val) is ShardedTensor:
        if len(val.local_shards()) == 0:
            return False
        if type(val.local_shards()[0].tensor) is ShardedTensor:
            return True
    return False

def _module_sharding_plan(specs):
    colwise_spec, rowwise_spec = specs[0], specs[1]
    return ShardingPlan(
        plan={
            "net1.weight": colwise_spec,
            "net2.weight": rowwise_spec,
        },
        output_plan={
            "net2": colwise_spec,
        },
        return_local_tensor=["net2"],
    )


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))


class TestTpFsdpIntegration(FSDPTest):
    def _params_fsdp_flat_order(self, m, params_sharded, tp_world_size):
        params = {}
        sharding_info = {}
        for name, param in m.named_parameters():
            if name not in params_sharded:
                params[name] = param.view(-1).size(0)
            else:
                params[name] = param.view(-1).size(0) // tp_world_size
                sharding_info[name] = (param.size(), 0 if "net1" in name else 1)
        return params, sharding_info

    # Generate the sub-process group for both TP and FSDP.
    # For example, if total world_size = 8 and tp_world_size = 2.
    # The default pg is [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # We will create four TP sub pgs: [0, 1], [2, 3], [4, 5], [6, 7]
    # and two pgs for FSDP: [0, 2, 4, 6] and [1, 3, 5, 7]
    def _generate_sub_pg(self, tp_world_size):
        tp_ids = []
        fsdp_ids = []
        for i in range(self.world_size):
            idx = i // tp_world_size
            if len(tp_ids) <= idx:
                tp_ids.append([])
            tp_ids[idx].append(i)
            idx = i % tp_world_size
            if len(fsdp_ids) <= idx:
                fsdp_ids.append([])
            fsdp_ids[idx].append(i)

        tp_pgs = [dist.new_group(ids) for ids in tp_ids]
        data_parallel_pgs = [dist.new_group(ids) for ids in fsdp_ids]
        tp_pg = tp_pgs[self.rank // tp_world_size]
        fsdp_pg = data_parallel_pgs[self.rank % tp_world_size]
        return tp_pg, fsdp_pg

    def _get_module_optim(self, module, tp_enabled) -> Optional[ShardedOptimizer]:
        if tp_enabled:
            return ShardedOptimizer(
                dict(named_params_with_sharded_tensor(module)),
                torch.optim.SGD,
                lr=LR,
            )

    def _collect_fsdp_params_grads(
        self,
        m,
        is_sharded,
        params_fsdp_flat_order,
        sharding_info,
        tp_pg,
        fsdp_pg,
    ):
        local_grads = (
            torch.cat([torch.flatten(p.grad) for p in m.parameters()])
            .contiguous()
            .cuda(self.rank)
        )
        all_grads = torch.cat(
            [torch.empty_like(local_grads) for _ in range(fsdp_pg.size())]
        ).contiguous()
        dist._all_gather_base(all_grads, local_grads, group=fsdp_pg)
        if not is_sharded:
            return all_grads
        splits = tuple(params_fsdp_flat_order.values())
        all_grads = list(all_grads.split(splits))
        # Reconstruct sharded parameters.
        for idx, key in enumerate(params_fsdp_flat_order.keys()):
            if key in SHARD_PARAMS:
                local_tensor_size = list(sharding_info[key][0])
                sharding_dim = sharding_info[key][1]
                local_tensor_size[sharding_dim] //= tp_pg.size()
                local_tensor = all_grads[idx].view(*local_tensor_size)
                local_tensors = [
                    torch.empty_like(local_tensor) for _ in range(tp_pg.size())
                ]
                dist.all_gather(local_tensors, local_tensor, group=tp_pg)
                all_grads[idx] = torch.cat(local_tensors, dim=sharding_dim).reshape(-1)
        return torch.cat(all_grads).contiguous()

    # Sync grads across ranks by using FSDP style average of gradients on all ranks.
    def _sync_tp_module_grads(self, m, tp_pg, params_fsdp_flat_order):
        fsdp_world_size = int(self.world_size // tp_pg.size())
        # TP handles gradients differently from FSDP. We need to divide by tp_pg size.
        for p in m.parameters():
            all_params = [torch.zeros_like(p) for _ in range(fsdp_world_size)]
            splits = tuple(params_fsdp_flat_order.values())
            all_params = torch.cat(all_params).contiguous().split(splits)
            for idx, key in enumerate(params_fsdp_flat_order.keys()):
                if key not in OPS_NOT_SHARD:
                    all_params[idx][:] = 1
            all_params = torch.cat(all_params).contiguous().type(torch.BoolTensor)
            cur_param = all_params.chunk(fsdp_world_size)[self.rank // tp_pg.size()]
            # We want to sync the layer 3 to make it same as FSDP only case.
            p_grad_device = p.grad.device
            p_grad = p.grad.clone().detach()
            p_grad = p_grad.cuda(self.rank)
            dist.all_reduce(p_grad, op=dist.ReduceOp.SUM, group=tp_pg)
            p_grad = p_grad.to(p_grad_device)
            p.grad[~cur_param] = p_grad[~cur_param]
            # Sharded Tensor add up all gradients, so we need to do average.
            p.grad /= tp_pg.size()

    @skip_if_lt_x_gpu(4)
    @parametrize("model_parallel_size", [2, 4])
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    def test_fsdp_tp_integration(self, model_parallel_size, cpu_offload):
        """Test FSDP with TP Integration."""
        inp_size = [2, 3, 5]
        # Use same seed so that each rank get the same model params.
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        torch.manual_seed(0)
        model_tp = SimpleModel().cuda(self.rank)
        params_fsdp_flat_order, sharding_info = self._params_fsdp_flat_order(
            model, SHARD_PARAMS, model_parallel_size
        )

        # Create Input
        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        input = torch.rand(*inp_size).cuda(self.rank)
        self.assertEqual(model(input), model_tp(input))

        # Generate sub-process group for TP and FSDP.
        tp_pg, fsdp_pg = self._generate_sub_pg(model_parallel_size)

        # Wrap the control group model with FSDP only.
        model = FSDP(model, process_group=self.process_group, cpu_offload=cpu_offload)
        sharding_specs = _generate_chunk_sharding_spec(tp_pg.size(), tp_pg)
        # Shard the module first and then wrap with FSDP.
        sharding_plan = _module_sharding_plan(sharding_specs)
        shard_module(model_tp, sharding_plan, process_group=tp_pg)

        model_tp = FSDP(model_tp, process_group=fsdp_pg, cpu_offload=cpu_offload)

        # Run first pass forward for both models
        output = model(input)
        output_tp = model_tp(input)
        self.assertEqual(output, output_tp)

        # Use a simple sum as loss to verify backward and grad computation.
        output.sum().backward()
        output_tp.sum().backward()

        # Sync gradient to ensure we have same gradient across different tp groups.
        self._sync_tp_module_grads(
            model_tp,
            tp_pg,
            params_fsdp_flat_order,
        )

        # Compare grads value.
        model_grads = self._collect_fsdp_params_grads(
            model,
            False,
            params_fsdp_flat_order,
            sharding_info,
            None,
            self.process_group,
        )
        model_tp_grads = self._collect_fsdp_params_grads(
            model_tp,
            True,
            params_fsdp_flat_order,
            sharding_info,
            tp_pg,
            fsdp_pg,
        )
        self.assertEqual(model_grads, model_tp_grads)

        # Run optimizer to update the params.
        model_optim = torch.optim.SGD(model.parameters(), lr=LR)
        model_tp_optim = torch.optim.SGD(model_tp.parameters(), lr=LR)
        model_optim.step()
        model_tp_optim.step()

        # Run second pass forward for both models after backward and optimization.
        torch.manual_seed(input_seed + 16)
        input = torch.rand(*inp_size).cuda(self.rank)
        output = model(input)
        output_tp = model_tp(input)
        self.assertEqual(output, output_tp)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_checkpoint_integration(self):
        """Test FSDP with TP Integration."""
        inp_size = [2, 3, 5]
        model_parallel_size = 2
        # Use same seed so that each rank get the same model params.
        torch.manual_seed(0)
        model_tp = SimpleModel().cuda(self.rank)
        # Generate sub-process group for TP and FSDP.
        tp_pg, fsdp_pg = self._generate_sub_pg(model_parallel_size)

        # Wrap the control group model with FSDP only.
        sharding_specs = _generate_chunk_sharding_spec(tp_pg.size(), tp_pg)
        # Shard the module first and then wrap with FSDP.
        sharding_plan = _module_sharding_plan(sharding_specs)
        shard_module(model_tp, sharding_plan, process_group=tp_pg)
        model_tp = FSDP(model_tp, process_group=fsdp_pg)

        # check we produce a nested ST
        with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
            state_dict = model_tp.state_dict()
            # TODO once 2D is out, validate the nesting
            self.assertTrue(_is_nested_tensor(state_dict["net1.weight"]))
            self.assertFalse(_is_nested_tensor(state_dict["net1.bias"]))

        optim_input = list(model_tp.parameters())
        optim = torch.optim.Adam(optim_input, lr=0.0001)

        # Create Input
        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        input = torch.rand(*inp_size).cuda(self.rank)

        model_tp(input).sum().backward()
        optim.step()

        optim_state = FSDP.sharded_optim_state_dict(model_tp, optim, optim_input)
        # TODO once 2D is out, validate the nesting
        self.assertTrue(_is_nested_tensor(optim_state["state"]["net1.weight"]["exp_avg"]))
        self.assertFalse(_is_nested_tensor(optim_state["state"]["net1.bias"]["exp_avg"]))


instantiate_parametrized_tests(TestTpFsdpIntegration)

if __name__ == "__main__":
    run_tests()
