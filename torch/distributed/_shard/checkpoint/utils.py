from typing import Any, List, Union
import torch.distributed as dist

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
