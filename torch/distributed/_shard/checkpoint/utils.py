import email
from typing import Any, List, Union, Callable, Tuple
import torch.distributed as dist
import traceback
from .api import CheckpointException
import torch

def tensor_narrow_n(tensor: torch.Tensor, offsets: Tuple[int, ...], lengths: Tuple[int, ...]) -> torch.Tensor:
    for dim, (start, length) in enumerate(zip(offsets, lengths)):
        tensor = torch.narrow(tensor, dim, start, length)
    return tensor

class DistWrapper:
    def __init__(self, group: dist.ProcessGroup, use_dist: bool, coordinator_rank: int):
        self.group = group
        self.use_dist = use_dist
        self.coordinator_rank = coordinator_rank
        if self.use_dist:
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.is_coordinator = True


    def get_rank(self) -> int:
        if self.use_dist:
            return dist.get_rank(self.group)
        return 0

    def get_world_size(self) -> int:
        if self.use_dist:
            return dist.get_world_size(self.group)
        return 1

    def broadcast(self, object: Any) -> Any:
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(
                object_list=object_list,
                group=self.group,
                src=self.coordinator_rank)
        return object_list[0]

    def gather(self, object: Any) -> Union[List[Any], None]:
        if self.use_dist:
            gather_objs = [None] * dist.get_world_size(self.group) if self.is_coordinator else None

            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.coordinator_rank,
                group=self.group
            )
            # flatten
            result = gather_objs
        else:
            result = [object]
        return result

    def allgather(self, object: Any) -> List[Any]:
        if self.use_dist:
            gather_objs = [None] * dist.get_world_size(self.group) if self.is_coordinator else None

            dist.all_gather_object(
                object_list=gather_objs,
                obj=object,
                group=self.group
            )
        else:
            gather_objs = [object]
        return gather_objs

    def scatter(self, object_list: List[Any]) -> Any:
        if self.use_dist:
            gather_result = [None]
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=object_list if self.is_coordinator else None,
                src=self.coordinator_rank,
                group=self.group
            )

            local_reply = gather_result[0]
        else:
            local_reply = object_list[0]
        return local_reply

    def run_on_coordinator(self, step, coordinator_cb: Callable[[], Any]) -> Any:
        result: Any
        if self.is_coordinator:
            try:
                result = coordinator_cb()
            except BaseException as e:
                traceback.print_exc()
                result = e

        # Writing can only start once prepare has finished
        exception = self.broadcast(result)
        if isinstance(exception, BaseException):
            raise CheckpointException(step, {self.coordinator_rank : exception})

    def map_scatter(self,
        step: str,
        map_cb: Callable[[], Any],
        coordinator_cb: Callable[[List[Any]], List[Any]]
    ) -> Tuple[Any, Any]:
        try:
            local_data = map_cb()
        except BaseException as e:
            traceback.print_exc()
            local_data = e

        all_data = self.gather(local_data)
        if self.is_coordinator:
            node_failures = {i: err for i, err in enumerate(all_data) if isinstance(err, BaseException)}

            if len(node_failures) == 0:
                try:
                    all_results = coordinator_cb(all_data)
                except BaseException as e:
                    traceback.print_exc()
                    node_failures[self.rank] = e
            
            if len(node_failures) > 0:
                all_results = [CheckpointException(step, node_failures)] * self.get_world_size()

        result = self.scatter(all_results)
        if isinstance(result, BaseException):
            raise result
        return result

    def map_reduce(self,
        step: str,
        map_cb: Callable[[], Any],
        reduce_cb: Callable[[List[Any]], Any]
    ) -> Tuple[Any, Any]:
        try:
            local_data = map_cb()
        except BaseException as e:
            traceback.print_exc()
            local_data = e

        all_data = self.gather(local_data)
        if self.is_coordinator:
            node_failures = {i: err for i, err in enumerate(all_data) if isinstance(err, BaseException)}

            if len(node_failures) == 0:
                try:
                    result = reduce_cb(all_data)
                except BaseException as e:
                    traceback.print_exc()
                    node_failures[self.rank] = e
            
            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        result = self.broadcast(result)
        if isinstance(result, BaseException):
            raise result
        return result

    def run_on_all_ranks(self,
        step: str,
        callback: Callable[[], Any],
    ) -> Tuple[Any, Any]:
        try:
            _ = callback()
            result = None
        except BaseException as e:
            traceback.print_exc()
            result = e

        all_results = self.allgather(result)

        node_failures = {i: err for i, err in enumerate(all_results) if isinstance(err, BaseException)}
        if len(node_failures) > 0:
            raise CheckpointException(step, node_failures)
