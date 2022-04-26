import abc
from typing import List, Dict

from torch.futures import Future

from .metadata import (
    BytesReadRequest,
    BytesWriteRequest,
    Metadata,
    TensorReadRequest,
    TensorWriteRequest,
)

class StorageWriter(abc.ABC):
    """
    Interface to write to underlying storage system
    """

    @abc.abstractmethod
    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        """
        Performs a write request and returns a Future to wait on.
        Args:
            requests (BytesWriteRequest): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        """
        Performs a write request and returns a Future to wait on.
        Args:
            requests (TensorWriteRequest): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def write_metadata(self, metadata: Metadata) -> None:
        """
        Writes the metatdata.

        Args:
            metadata (Metadata): see `./metadata.py`
        """
        pass

    def prepare_storage(self, storage_keys: Dict[str, int]) -> None:
        """
        This blocking call can be overwritten by the subclass.
        It can use `storage_keys` to plan for any write preformace optimization.
        e.g. non sequential and parallel writes.
        By default, it does nothing

        Args:
            storage_keys (Dict[str, int]): key - handle's name. value - size
                of the handle.
        """
        pass


class StorageReader(abc.ABC):
    """
    Interface to read from the underlying storage system.
    """

    @abc.abstractmethod
    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        """
        Read request and returns a Future to wait on.
        Args:
            requests (List[BytesReadRequest]): see `./metadata.py`]
        """
        pass

    @abc.abstractmethod
    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Performs a read request and returns a Future to wait on.
        Args:
            requests (List[BytesReadRequest]): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Read the meta data and returns.
        """
        pass
