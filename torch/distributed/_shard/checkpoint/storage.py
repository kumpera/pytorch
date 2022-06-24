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
    index: MetadataIndex
    type: LoadItemType

    # Offset from tensor found in checkpoint metadata
    storage_offsets: torch.Size
    # Offset from stored tensor
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

    A StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    1) (all ranks) init()
    1) (coordinator) prepare()
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
            is_coordinator (bool): Whether this instance is reponsible for coordinator all writes
        """
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Add storage specific data to the plan.

        While this method can produce a completely different plan, the prefered
        way is to store storage specific data in SavePlan::storage_data and WriteItem::storage_data.
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """
        Set storage specific data to the global plan.


        While this method can produce a completely different plan, the prefered
        way is to store storage specific data in SavePlan::storage_data and WriteItem::storage_data.
        """
        pass


    @abc.abstractmethod
    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner
    ) -> Future[List[WriteResult]]:
        """
        Initiate writes for all requests in `requests`.

        Writing can happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Implementors are responsible for any device to host transfers required
        to copy.

        Args:
            requests (List[TensorWriteRequest]): A list of requests to write

        Returns:
            A future that completes once all writes have finished.
        """
        pass

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        Writes the metadata and marks the current checkpoint as sucessfull.

        This method is called once globally after all data was writen
        and is used to write its metadata and commit the checkpoint.

        The `metadata` object includes a global view of the checkpoint
        and, while writing it is optional, it must be recoverable by the
        StorageReader implementation.

        The actual format/schema used for serializing `metadata` is
        considered and implementation detail.

        Args:
            metadata (Metadata): metadata for the new checkpoint

        Returns:
            None
        """
        pass

class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    A subclass should expected the following sequence of calls by ``load_state_dict``:

    1) read_metadata() - on all ranks
    2) read_bytes
    3) read_tensors

    The reads from (2) and (3) are initiated before any waiting is done.

    Implementors must ensure host/device synchronization as part of
    completion of both read requests.
    """
    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Reads the checkpoint metadata.

        Returnss:
            The metatada object associated with the checkpoint being loaded.

        """
        pass

    @abc.abstractmethod
    def init(self, metadata: Metadata) -> None:
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        pass

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        pass
