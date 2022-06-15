import itertools
from dataclasses import dataclass
import os
import io
import pickle
from typing import List, Optional, Union, Any, Tuple, Dict, cast

import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path

from .metadata import (
    Metadata,
)
from .storage import (
    LocalPlan,
    LoadPlan,
    StorageReader,
    StorageWriter,
    WriteResult,
    BytesReadRequest,
    BytesWriteRequest,
    TensorReadRequest,
    TensorWriteRequest,
    RESOLVE_DATA_TYPE
)

from .resharding import (
    default_prepare_writes,
    default_prepare_reads
)

from .utils import tensor_narrow_n

@dataclass
class _StorageInfo:
    relative_path: str
    offset: int
    length: int

@dataclass
class _StoragePrefix:
    prefix: str


class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    def __init__(self, path: Union[str, os.PathLike], single_file_per_rank: bool = False) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
        """
        super().__init__()
        self.path = Path(path)
        self.single_file_per_rank = single_file_per_rank

    def prepare_local_plan(self, plan: LocalPlan) -> LocalPlan:
        # There's no storage input in the local plan
        return plan

    def prepare_global_plan(self, global_plan: List[LocalPlan]) -> List[LocalPlan]:
        # Add a prefix for each rank
        # FIXME maybe make this the default behavior?
        for i, plan in enumerate(global_plan):
            plan.storage_data = _StoragePrefix(f"__{i}_")
        return global_plan

    def prepare_writes(
        self,
        state_dict: Dict[str, Any],
        plan: LocalPlan,
        resolve_data: RESOLVE_DATA_TYPE,
    ) -> Tuple[List[TensorWriteRequest], List[BytesWriteRequest]]:
        return default_prepare_writes(state_dict, plan, resolve_data)

    def write_data(
        self,
        storage_plan: _StoragePrefix,
        tensors: List[TensorWriteRequest],
        bytes: List[BytesWriteRequest]
    ) -> Future[List[WriteResult]]:
        # The following couple lines are simple implementation to get
        # things going.
        #
        # At load time, to enable resharding, we use (sub)view of the tensor.
        # Since the storage of the tensor might not be contiguous. we need to
        # preserve the original view, to calculate the correct sub view at load.
        #
        # `torch.save` saves both the view and storage, it is a good option
        # for unblocking. There are two drawbacks:
        # 1. `torch.save` is pickle based, and pickle is not known for its
        #   compatibility, we should consider replacing it with a more
        #   stable option.
        # 2. pickle is not streamable.
        res = []
        file_count = 0

        def gen_file(prefix: _StoragePrefix):
            nonlocal file_count
            file_name = f"{prefix.prefix}{file_count}"
            file_count += 1
            return file_name

        # This is uggly, cleanup
        if self.single_file_per_rank:
            file_name = gen_file(storage_plan)
            with (self.path / file_name).open("wb") as w:
                for tensor_req in tensors:
                    offset = w.tell()
                    torch.save(tensor_req.tensor, w)
                    length = w.tell() - offset
                    res.append(WriteResult.from_write_item(
                        tensor_req.item,
                        _StorageInfo(file_name, offset, length)
                    ))

                for bytes_req in bytes:
                    offset = w.tell()
                    w.write(bytes_req.bytes.getbuffer())
                    length = w.tell() - offset
                    res.append(WriteResult.from_write_item(
                        bytes_req.item,
                        _StorageInfo(file_name, offset, length),
                    ))

                os.fsync(w.fileno())

        else:
            for tensor_req in tensors:
                file_name = gen_file(storage_plan)
                with (self.path / file_name).open("wb") as w:
                    torch.save(tensor_req.tensor, w)
                    os.fsync(w.fileno())
                    res.append(WriteResult.from_write_item(
                        tensor_req.item,
                        _StorageInfo(file_name, 0, w.tell())
                    ))

            for bytes_req in bytes:
                file_name = gen_file(storage_plan)
                with (self.path / file_name).open("wb") as w:
                    w.write(bytes_req.bytes.getbuffer())
                    os.fsync(w.fileno())
                    res.append(WriteResult.from_write_item(
                        bytes_req.item,
                        _StorageInfo(file_name, 0, w.tell()),
                    ))

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(res)
        return fut

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def finish(self, metadata: Metadata) -> None:
        with (self.path / ".metadata.tmp").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")


class SlicedBufferedReader(io.BufferedReader):
    def __init__(self, base_stream: io.RawIOBase, offset: int, len: int):
        super().__init__(base_stream)
        self.offset = offset
        self.len = len
        self.seek(0)

    def seek(self, __offset: int, __whence: int = os.SEEK_SET) -> int:
        if __whence == os.SEEK_SET:
            __offset = self.offset + __offset
        elif __whence == os.SEEK_END:
            __whence = os.SEEK_SET
            __offset = (self.offset + self.len) - __offset
        return super().seek(__offset, __whence)

    def tell(self) -> int:
        return super().tell() - self.offset

class FileSystemReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = Path(path)

    def _slice_file(self, file, sinfo: _StorageInfo):
        return SlicedBufferedReader(
            io.FileIO(file.fileno(),closefd=False), 
            sinfo.offset, sinfo.length
        )

    def read_data(self,
        storage_plan: Any,
        byte_requests: List[BytesReadRequest],
        tensor_requests: List[TensorReadRequest]
    ) -> Future[None]:

        # group requests by file
        per_file = dict()
        for br in itertools.chain(byte_requests, tensor_requests):
            path = br.meta.storage_data.relative_path
            per_file.setdefault(path, []).append(br)

        for relative_path, reqs in per_file.items():
            with (self.path / relative_path).open("rb") as file:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    file_slice = self._slice_file(file, req.meta.storage_data)
                    
                    if isinstance(req, TensorReadRequest):
                        view_to_copy = cast(Tensor, torch.load(file_slice, map_location=req.target_device))
                        view_to_copy = tensor_narrow_n(view_to_copy, req.storage_offsets, req.lengths)
                        req.copy(view_to_copy)
                    else:
                        req.copy(torch.load(file_slice))

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with (self.path / ".metadata").open("rb") as metadata_file:
            return pickle.load(metadata_file)

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan

    def prepare_reads(
        state_dict: Dict[str, Any],
        load_plan: LoadPlan,
        load_bytes_callback,
        copy_tensor_callback,
    ) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
        return default_prepare_reads(
            state_dict,
            load_plan,
            load_bytes_callback,
            copy_tensor_callback,
        )
