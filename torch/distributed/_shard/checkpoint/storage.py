import abc
from dataclasses import dataclass
from typing import List, Union, Any, Dict, Callable, Tuple, Optional, cast
from torch.distributed._shard.sharded_tensor import ShardedTensor

import torch
import io
from torch.futures import Future
from enum import Enum, auto

from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorInfo,
    TensorStorageMetadata,
)

from torch.distributed._shard.sharded_tensor import (
    ShardMetadata
)
""""
Notes on design issues:

We need to resurface the prepare_xxx_ functions
Can we unify WriteItem with ReadItem? Can we unity LocalPlan / LoadPlan?
How about the planning phase?
The way shard MD is defined in WriteItem is bad.

There's a layer complexity/confusion going on the read side:
On the write side, the planner picks stuff from the state_dict and handle that to the storage layer to figure out how to write each of them.

On the read side, there's little for the planner itself to do beyond handling any transforms done by the write planner.

What sort of scenarios need a central read plan?
    One case that's reasonable but hard to deal with is to reduce read applification
    when resharding.

There's this odd separation of how we deserialize BytesIO and tensors.
    Tensor is done by storage
    bytesIO is done by Planner
-> Fixed, all on storage

Where should tensor narrowing happen?
    Dest tensor happens in storage::prepare_reads
    Storage tensor happens in storage::write_data

We ignore lenghts/offset for BytesIO and this is something we wanna care

I'm not 100% sure of prepare_reads / prepare_writes WRT passing of callbacks.
    Maybe pass a structurally typed object that has those methods?

The current design assumes storage / planner to be stateless objects.
    Having an init method on both that we pass some global data to them.
        Like state_dict and metadata.

"""


class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()

@dataclass
class WriteItem:
    #this is the FQN in the metadata
    fqn: str
    type: WriteItemType

    # Next two valid if this is a tensor write
    chunk: Optional[ChunkStorageMetadata] = None
    tensor_info: Optional[TensorInfo] = None
    # This is the index into Metadata
    chunk_index: Optional[int] = None

    planner_data: Any = None
    storage_data: Any = None

    @property
    def is_tensor(self):
        return self.type == WriteItemType.TENSOR

    @property
    def is_bytesio(self):
        return self.type == WriteItemType.BYTE_IO

    @property
    def is_shard(self):
        return self.type == WriteItemType.SHARD


@dataclass
class WriteResult:
    fqn: str
    # For tensor writes
    chunk_index: Optional[int]

    size_in_bytes: int
    planner_data: Any
    storage_data: Any

@dataclass
class LocalPlan:
    items: List[WriteItem]
    storage_data: Any = None
    planner_data: Any = None

@dataclass
class ReadItem:
    fqn: str

    # Offset from tensor found in checkpoint metadata
    storage_offsets: torch.Size
    # Offset from stored tensor
    dest_offsets: torch.Size
    lengths: torch.Size

    chunk: Optional[ChunkStorageMetadata] = None
    chunk_index: Optional[int] = None

    planner_data: Any = None
    storage_data: Any = None

    @property
    def is_tensor(self):
        return self.chunk is not None

    @property
    def is_bytesio(self):
        return self.chunk is None

STATE_DICT_TYPE = Dict[str, Any]

class SavePlanner(abc.ABC):
    @abc.abstractmethod
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        """
        Intialize this planner to save ``state_dict``.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> LocalPlan:
        """
        Compute the save plan for the current rank. This will be aggregated and fed into create_global_plan so any inputs for global planning should be returned from here.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, all_plans: List[LocalPlan]) -> List[LocalPlan]:
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, new_plan: LocalPlan) -> LocalPlan:
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
    def write_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        pass

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        pass


class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    A subclass should expect the following sequence of calls by ``save_state_dict``

    1) (called once globally) prepare()
    2) prepare_local_plan()
    3) (called once globally) prepare_global_plan.
    4) prepare_writes()
    5) write_data()
    6) (called once globally) finish().

    There's a single process that executes methods that are called once globally.
    """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LocalPlan) -> LocalPlan:
        """
        Add storage specific data to the plan.

        While this method can produce a completely different plan, the prefered
        way is to store storage specific data in LocalPlan::storage_data and WriteItem::storage_data.
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LocalPlan]) -> List[LocalPlan]:
        """
        Set storage specific data to the global plan.


        While this method can produce a completely different plan, the prefered
        way is to store storage specific data in LocalPlan::storage_data and WriteItem::storage_data.
        """
        pass

    @abc.abstractmethod
    def prepare(self) -> None:
        """
        Initialize storage to receive the checkpoint.

        This method is called once globally per checkpoint before any other method.
        This is in contrast to ``prepare_storage`` which is called on each process
        in parallel.

        Returns:
            Future to signal intialization is complete.
        """
        pass

    @abc.abstractmethod
    def write_data(
        self,
        plan: LocalPlan,
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
    def finish(self, metadata: Metadata) -> None:
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
    def prepare_local_plan(self, metadata: Metadata, plan: LoadPlan) -> LoadPlan:
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        pass

    @abc.abstractmethod
    def read_data(self,
        plan: LoadPlan,
        planner: LoadPlanner
    ) -> Future[None]:
        pass

