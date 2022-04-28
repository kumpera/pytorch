import os
import operator
import pickle
from typing import List, Optional, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.futures import Future
from pathlib import Path

from .metadata import (
    BytesReadRequest,
    BytesWriteRequest,
    Metadata,
    TensorReadRequest,
    TensorWriteRequest,
)
from .storage import StorageReader, StorageWriter

def _get_file_storage_path(
    storage_path: str,
    storage_key: str
) -> Path:
    return (Path(storage_path) / storage_key)

class FileSystemWriter(StorageWriter):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        for req in requests:
            _get_file_storage_path(self.path, req.storage_key).write_bytes(
                req.bytes.getbuffer()
            )
        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        for req in requests:
            # The following couple lines are simple implementation to get
            # things going.
            #
            # At load time, to enable resharding, we use (sub)view of the tensor.
            # Since the storage of the tensor might not be contiguous. we need to
            # preseve the original view, to calculate the correct sub view at load.
            #
            # `torch.save` saves both the view and storage, it is a good option
            # for unblocking. There are two drawbacks:
            # 1. `torch.save` is pickle based, and pickle is not known for its
            #   compatibility, we should consider replacing it with a more
            #   stable option.
            # 2. pickle is not streamable.
            with _get_file_storage_path(self.path, req.storage_key).open("wb") as w:
                torch.save(req.tensor, w)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in Storage Writer
    def write_metadata(self, metadata: Metadata) -> None:
        # Once write metadata once as each Metadata has the global view
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        with _get_file_storage_path(self.path, ".metadata").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)

class FileSystemReader(StorageReader):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Very basic implementation that read from file system.
        """
        # Sort the the requests by storage key and try to reuse the loaded tensors
        requests.sort(key=operator.attrgetter("storage_key"))

        cached_storage_key = None
        view_cached: Optional[Tensor] = None

        for req in requests:
            if cached_storage_key != req.storage_key or \
                    (view_cached is not None and view_cached.device != req.tensor.device):

                with _get_file_storage_path(self.path, req.storage_key).open("rb") as storage:
                    view_cached = cast(Tensor, torch.load(storage, map_location=req.tensor.device))
                    cached_storage_key = req.storage_key

            view_to_copy: Tensor = cast(Tensor, view_cached)
            # FileSystemWrite writes the tensor as is during save.
            # During load time, we will load the Tensor (with it orignal view)
            # narrow it along all dimemsions, and copy_ it to the
            # target tensor, which will be the same size.
            for dim, (start, length) in enumerate(zip(req.offsets, req.lengths)):
                view_to_copy = torch.narrow(view_to_copy, dim, start, length)

            assert (
                view_to_copy.size() == req.tensor.size()
            ), f"The {req.storage_key} src/dst size does not match."


            assert (
                view_to_copy.device == req.tensor.device
            ), f"cannot load across devices {view_to_copy.device} vs {req.tensor.device}"

            req.tensor.copy_(view_to_copy)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        for req in requests:
            with _get_file_storage_path(self.path, req.storage_key).open("rb") as storage:
                req.bytes.write(storage.read())

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with _get_file_storage_path(self.path, ".metadata").open("rb") as metadata_file:
            return pickle.load(metadata_file)
