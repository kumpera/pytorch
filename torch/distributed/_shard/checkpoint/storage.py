import abc
from dataclasses import dataclass
from typing import List, Union, Any, Dict, Callable, Tuple, Optional, cast

import torch
import io
from torch.futures import Future

from .metadata import (
    BytesStorageMetadata,
    Metadata,
    BytesIOProperties,
    ShardStorageMetadata,
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


@dataclass
class WriteItem:
    fqn: str
    meta: Union[Tuple[ShardMetadata, TensorInfo], TensorInfo, BytesIOProperties]
    storage_data: Any = None
    planner_data: Any = None

    @property
    def is_shard(self):
        return isinstance(self.meta, Tuple)

    @property
    def is_tensor(self):
        return isinstance(self.meta, TensorInfo)

    @property
    def is_bytesio(self):
        return isinstance(self.meta, BytesIOProperties)

    def lookup(self, state_dict: Dict[str, Any]) -> Any:
        obj = state_dict[self.fqn]
        if not self.is_shard:
            return obj

        for shard in obj.local_shards():
            if shard.metadata == self.meta[0]:
                return shard.tensor
        raise ValueError(f"could not find shard '{self.meta[0]}' for FQN: '{self.fqn}'")

@dataclass
class LocalPlan:
    items: List[WriteItem]
    storage_data: Any
    planner_data: Any


class ReadItem:
    fqn: str
    meta: Union[BytesStorageMetadata, ShardStorageMetadata, TensorStorageMetadata]

    # FIXME this sort of imply 

    # Offset from tensor found in checkpoint metadata
    storage_offsets: Tuple[int, ...]
    # Offset from stored tensor
    dest_offsets: Tuple[int, ...]
    lengths: Tuple[int, ...]

    storage_data: Any = None
    planner_data: Any = None

    @property
    def is_shard(self):
        return isinstance(self.meta, ShardStorageMetadata)

    @property
    def is_tensor(self):
        return isinstance(self.meta, TensorStorageMetadata)

    @property
    def is_bytesio(self):
        return isinstance(self.meta, BytesStorageMetadata)

    def lookup(self, state_dict: Dict[str, Any]) -> Any:
        obj = state_dict[self.fqn]
        if not self.is_shard:
            return obj

        for shard in obj.local_shards():
            if shard.metadata == cast(ShardStorageMetadata, self.meta).shard_metadata:
                return shard.tensor
        raise ValueError(f"could not find shard '{self.meta[0]}' for FQN: '{self.fqn}'")

STATE_DICT_TYPE = Dict[str, Any]
RESOLVE_WI_TYPE = Callable[[WriteItem], Union[torch.Tensor, io.BytesIO]]
RESOLVE_DATA_TYPE = Callable[[STATE_DICT_TYPE, WriteItem], Union[torch.Tensor, io.BytesIO]]


class TensorWriteRequest:
    def __init__(self, item: WriteItem, resolve_data: RESOLVE_WI_TYPE):
        self._item = item
        self._resolve = resolve_data
        self._obj = None

    @property
    def item(self) -> WriteItem:
        return self._item

    @property
    def tensor(self) -> torch.Tensor:
        if self._obj is None:
            self._obj = self._resolve(self._item)
        return cast(torch.Tensor, self._obj)

@dataclass
class BytesWriteRequest:
    def __init__(self, item: WriteItem, resolve_data: RESOLVE_WI_TYPE):
        self._item = item
        self._resolve = resolve_data
        self._obj = None

    @property
    def item(self) -> WriteItem:
        return self._item

    @property
    def bytes(self) -> io.BytesIO:
        if self._obj is None:
            self._obj = self._resolve(self._item)
        return cast(io.BytesIO, self._obj)

@dataclass
class BytesReadRequest:
    def __init__(
        self,
        item: ReadItem,
        copy
    ):
        self._item = item
        self._copy = copy

    # @property
    # def fqn(self) -> str:
    #     return self.item.fqn

    @property
    def meta(self) -> BytesStorageMetadata:
        return self._item.meta

    def copy(self, object) -> Future[None]:
        return self._copy(self._item, object)

RESOLVE_TENSOR_TYPE = Callable[[torch.Tensor], torch.Tensor]
COPY_TENSOR_TYPE = Callable[[torch.Tensor, torch.Tensor], Future[None]]

class TensorReadRequest:
    def __init__(
        self,
        item: ReadItem,
        tensor: torch.Tensor,
        copy

        # fqn: str,
        # meta: Union[TensorStorageMetadata, ShardStorageMetadata],
        # tensor: torch.Tensor,
        # offsets: Tuple[int, ...],
        # lengths: Tuple[int, ...],
        # resolve: RESOLVE_TENSOR_TYPE,
        # copy: COPY_TENSOR_TYPE
    ):
        self._item = item
        self._tensor = tensor
        self._copy = copy

    # @property
    # def fqn(self) -> str:
    #     return self.item.fqn

    # @property
    # def offsets(self) -> str:
    #     return self._offsets

    # @property
    # def lengths(self) -> str:
    #     return self._lengths

    @property
    def meta(self) -> Union[TensorStorageMetadata, ShardStorageMetadata]:
        return self._item.meta
    @property
    def target_device(self) -> torch.device:
        return self._tensor.device

    # def resolve(self, tensor: torch.Tensor) -> torch.Tensor:
    #     return self._resolve(tensor)

    def copy(self, data: torch.Tensor) -> None:
        # FIXME maybe this has no place here
        assert (
            data.size() == self._tensor.size()
        ), f"req {self._fqn} mismatch sizes {data.size()} vs {self._tensor.size()}"

        assert (
            data.device == self._tensor.device
        ), f"req {self.fqn} mismatch devices: {data.device} vs {self._tensor.target_device}"

        self._copy(self._tensor, data)

@dataclass
class WriteResult:
    fqn: str
    meta: Union[Tuple[ShardMetadata, TensorInfo], TensorInfo, BytesIOProperties]
    planner_data: Any
    storage_data: Any

    @classmethod
    def from_write_item(cls, item: WriteItem, storage: Any):
        return WriteResult(item.fqn, item.meta, item.planner_data, storage)

class Planner(abc.ABC):
    @abc.abstractmethod
    def create_local_plan(self, state_dict: Dict[str, Any], is_coordinator: bool) -> LocalPlan:
        """
        Compute the save plan for the current rank. This will be aggregated and fed into create_global_plan
        so any inputs for global planning should be returned from here.

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
    def merge_plans(self, original_plan: LocalPlan, new_plan: LocalPlan) -> LocalPlan:
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
    def resolve_data(self, state_dict: Dict[str, Any], write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
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
    def create_local_plan(self, state_dict, metadata: Metadata) -> LoadPlan:
        pass

    @abc.abstractmethod
    def create_global_plan(self, globla_plan: List[LoadPlan]) -> List[LoadPlan]:
        pass

    @abc.abstractmethod
    def merge_plans(self, original_plan: LoadPlan, new_plan: LoadPlan) -> LoadPlan:
        pass

    @abc.abstractmethod
    def load_bytes(self, state_dict, wi, stream) -> None:
        pass

    @abc.abstractmethod
    def copy_tensor(self, wi, dest: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
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
    def prepare_writes(
        self,
        state_dict: Dict[str, Any],
        plan: LocalPlan,
        resolve_data: RESOLVE_DATA_TYPE,
    ) -> Tuple[List[TensorWriteRequest], List[BytesWriteRequest]]:
        """
        Produce writes to satisfy those requested in ``plan``.
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
        storage_plan: Any,
        tensors: List[TensorWriteRequest],
        bytes: List[BytesWriteRequest]
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

    def prepare_storage(self, storage_writes: List[Union[TensorWriteRequest, BytesWriteRequest]]) -> None:
        """
        Prepare the underlying storage for upcoming writes.

        This is an optional override intended for advanced scenarios where
        a storage layer needs wants to do some work ahead of the writing itself.

        This method is called on each process in parallel before any writes are performed.

        The default implementation does nothing.

        Args:
            storage_writes (List[Union[TensorWriteRequest, BytesWriteRequest]]): A list of
            all writes that will be submited.

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
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        pass

    @abc.abstractmethod
    def prepare_reads(
        state_dict: Dict[str, Any],
        load_plan: LoadPlan,
    ) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
        pass

    @abc.abstractmethod
    def read_data(self,
        storage_plan: Any,
        byte_requests: List[BytesReadRequest],
        tensor_requests: List[TensorReadRequest]
    ) -> Future[None]:
        pass

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Reads the checkpoint metadata.

        Returnss:
            The metatada object associated with the checkpoint being loaded.

        """
        pass
