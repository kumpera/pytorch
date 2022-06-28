import abc
from dataclasses import dataclass
from typing import List, Union, Any, Dict, Optional

import torch
import io
from torch.futures import Future
from enum import Enum, auto

from .metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorInfo,
)

class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()

class LoadItemType(Enum):
    TENSOR = auto()
    BYTE_IO = auto()

@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata
    info: TensorInfo

@dataclass(frozen=True)
class WriteItem:
    index: MetadataIndex
    type: WriteItemType

    # Value present if it's a tensor write
    tensor_data: Optional[TensorWriteData] = None

@dataclass(frozen=True)
class WriteResult:
    index: MetadataIndex

    size_in_bytes: int
    storage_data: Any

@dataclass(frozen=True)
class SavePlan:
    items: List[WriteItem]
    storage_data: Any = None
    planner_data: Any = None

@dataclass(frozen=True)
class ReadItem:
    # this is an index into the checkpoint metadata
    index: MetadataIndex
    type: LoadItemType

    # Offset from tensor found in checkpoint metadata
    storage_offsets: torch.Size

    # index to lookup the destination tensor
    dest_index: MetadataIndex
    # Offset into the destination tensor
    dest_offsets: torch.Size
    lengths: torch.Size

STATE_DICT_TYPE = Dict[str, Any]

class SavePlanner(abc.ABC):
    @abc.abstractmethod
    def init(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
        """
        Intialize this planner to save ``state_dict``.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan:
        """
        Compute the save plan for the current rank.
        This will be aggregated and passed to create_global_plan.
        Planner specific data can be passed through SavePlan::planner_data.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, all_plans: List[SavePlan]) -> List[SavePlan]:
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """
        Merge the plan created by `create_local_plan` and the result of `create_global_plan`.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_checkpoint_metadata(self, all_results: List[List[WriteResult]]) -> Metadata:
        """
        Create the checkpoint global metadata. This is usually just aggregating all results.
        """

        pass

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        """
        Lookup the object associated with ``write_item``in `state_dict` and apply any
        transformation (such as serialization) prior to the IO layer consuming it.
        """
        pass

@dataclass
class LoadPlan:
    items: List[ReadItem]
    storage_data: Any = None
    planner_data: Any = None


class LoadPlanner:
    @abc.abstractmethod
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        """
        Initialize this instance to load data into ``state_dict``

        . N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        """
        Create a LoadPlan based on state_dict and metadata provided by init.

        . N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, globla_plan: List[LoadPlan]) -> List[LoadPlan]:
        """
        Compute the global load plan and return plans for each rank.

        . N.B. This is called on the coordinator rank only
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
        """
        Accept the plan from coordinator and return final LoadPlan.
        """
        pass

    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        pass

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        pass


class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    1) (all ranks) init()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """

    @abc.abstractmethod
    def init(self, is_coordinator: bool) -> None:
        """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is reponsible for coordinating
              the checkpoint.
        """
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recomended
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plan (SavePlan): The local plan from the ``SavePlanner`` in use.

        Returns:
            A transformed ``SavePlan`` after storage local planning
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """
        Perform centralized planning of storage.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the prefered
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plans: A list of ``SavePlan`` instances, one for each rank.

        Returns:
            A list of transformed ``SavePlan`` after storage global planning
        """
        pass

    @abc.abstractmethod
    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner
    ) -> Future[List[WriteResult]]:
        """
        Write all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``SavePlanner::resolve_data`` on each item
        from the plan to get access to the underlying object to write.

        Subclasses should lazily call `resolve_data` as it can allocate memory.
        In case of tensors, make following assuptions:

        - They might be on any device, including not matching the one on ``WriteItem::tensor_data``
        - They might be views or not contiguous. Only the projection needs to be saved.

        Args:
            plan (SavePlan): The save plan to execute.
            planner (SavePlanner): Planner object to be used to resolve items to data.

        Returns:
            A future that completes to a list of WriteResult
        """
        pass

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        Writes the metadata and marks the current checkpoint as sucessful.

        The actual format/schema used for serializing `metadata` is an
        implemetation detail. The only requirement is that it's recoverable
        in to the same object graph.

        Args:
            metadata (Metadata): metadata for the new checkpoint
            results: A list of WriteResults from all ranks.

        Returns:
            None
        """
        pass

class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    One StorageReader instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expected the following sequence of calls by ``load_state_dict``:

    1) (all ranks) read_metadata()
    2) (all ranks) init
    3) (all ranks) prepare_local_plan
    4) (coordinator) prepare_global_plan
    5) (all ranks) read_data
    """
    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Reads the checkpoint metadata.

        Returns:
            The metatada object associated with the checkpoint being loaded.

        """
        pass

    @abc.abstractmethod
    def init(self, metadata: Metadata, is_coordinator: bool) -> None:
        """
        Initialize this instance.

        Args:
            metadata (Metadata): The metadata schema to use.
            is_coordinator (bool): Whether this instance is reponsible for coordinating
              the checkpoint.
        """
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recomended
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plan (LoadPlan): The local plan from the ``LoadPlan`` in use.

        Returns:
            A transformed ``LoadPlan`` after storage local planning
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        """
        Perform centralized planning of storage loading.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the prefered
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plans: A list of ``LoadPlan`` instances, one for each rank.

        Returns:
            A list of transformed ``LoadPlan`` after storage global planning
        """
        pass

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Reads all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``LoadPlanner::load_bytes`` to deserialize a BytesIO
        object into the right place.

        A subclass should call ``LoadPlanner::resolve_tensor`` to get access to the
        tensors that in should load data into.

        It's the StorageLayer responsibility to properly schedule any cross device copies
        required.

        Args:
            plan (LoadPlan): The local plan to execute on
            planner (LoadPlanner): The planner object to use to resolve items.

        Returns:
            A future that completes once all reads are finished.
        """
        pass
