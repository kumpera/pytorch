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
        fqn: str,
        meta: BytesStorageMetadata,
        copy
    ):
        self._fqn = fqn
        self._meta = meta
        self._copy = copy

    @property
    def fqn(self) -> str:
        return self._fqn

    @property
    def meta(self) -> BytesStorageMetadata:
        return self._meta

    def copy(self, stream) -> Future[None]:
        return self._copy(stream)

RESOLVE_TENSOR_TYPE = Callable[[torch.Tensor], torch.Tensor]
COPY_TENSOR_TYPE = Callable[[torch.Tensor, torch.Tensor], Future[None]]

class TensorReadRequest:
    def __init__(
        self,
        fqn: str,
        meta: Union[TensorStorageMetadata, ShardStorageMetadata],
        tensor: torch.Tensor,
        offsets: Tuple[int, ...],
        lengths: Tuple[int, ...],
        resolve: RESOLVE_TENSOR_TYPE,
        copy: COPY_TENSOR_TYPE
    ):
        self._fqn = fqn
        self._meta = meta
        self._tensor = tensor
        self._offsets = offsets
        self._lengths = lengths
        self._resolve = resolve
        self._copy = copy

    @property
    def fqn(self) -> str:
        return self._fqn

    @property
    def offsets(self) -> str:
        return self._offsets

    @property
    def lengths(self) -> str:
        return self._lengths

    @property
    def meta(self) -> Union[TensorStorageMetadata, ShardStorageMetadata]:
        return self._meta

    @property
    def target_device(self) -> torch.device:
        return self._tensor.device

    def resolve(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._resolve(tensor)

    def copy(self, data: torch.Tensor) -> Optional[Future[None]]:
        # FIXME maybe this has no place here
        assert (
            data.size() == self._tensor.size()
        ), f"req {self._fqn} mismatch sizes {data.size()} vs {self._tensor.size()}"

        assert (
            data.device == self._tensor.device
        ), f"req {self.fqn} mismatch devices: {data.device} vs {self._tensor.target_device}"

        return self._copy(self._tensor, data)

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

class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    A subclass should expect the following sequence of calls by ``save_state_dict``

    1) (called once globally) prepare()
    2) prepare_storage() with the writes that will be used with (3) and (4).
    3) write_bytes
    4) write_tensors.
    5) Wait for (2) and (3) futures. If either fail, abort checkpoint.
    6) (called once globally) finish().

    There's a single process that executes methods that are called once globally.
    The writes from (3) and (4) are initiated before any waiting is done.
    The last call to finish() has the semantics of commiting the checkpoint.
    """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LocalPlan) -> LocalPlan:
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LocalPlan]) -> List[LocalPlan]:
        pass

    @abc.abstractmethod
    def prepare_writes(
        self,
        state_dict: Dict[str, Any],
        plan: LocalPlan,
        resolve_data: RESOLVE_DATA_TYPE,
    ) -> Tuple[List[TensorWriteRequest], List[BytesWriteRequest]]:
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
    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        """
        Initiate read for all requests in `requests`.

        Reading happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Args:
            requests (List[BytesReadRequest]): A list of requests to read.

        Return:
            A future that completes once all read have finished.
        """
        pass

    @abc.abstractmethod
    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Initiate read for all requests in `requests`.

        Reading happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Implementors must not assume that the original device
        at write time will be the same at read time.

        If an implementation uses asynchronous copies to device, it must
        ensure proper synchronization W.R.T. the returned future.

        Args:
            requests (List[BytesReadRequest]): A list of requests to read.

        Returns:
            A future that completes once all read have finished.
        """
        pass

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Reads the checkpoint metadata.

        Returnss:
            The metatada object associated with the checkpoint being loaded.

        """
        pass
