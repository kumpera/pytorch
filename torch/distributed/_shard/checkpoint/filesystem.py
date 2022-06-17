from dataclasses import dataclass
import os
import io
import pickle
from typing import List, Union, Dict, cast

import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path

from .metadata import (
    Metadata,
)
from .storage import (
    LoadPlanner,
    LoadPlan,
    SavePlan,
    SavePlanner,
    StorageReader,
    StorageWriter,
    WriteResult,
    ReadItem,
    WriteItem
)

from .utils import tensor_narrow_n


@dataclass
class _StorageInfo:
    """
    This is the per entry storage info
    """
    relative_path: str
    offset: int
    length: int

@dataclass
class _StoragePrefix:
    prefix: str


def result_from_write_item(item: WriteItem, size_in_bytes, storage_data) -> WriteResult:
    return WriteResult(
        fqn=item.fqn,
        chunk_index=item.chunk_index,
        size_in_bytes=size_in_bytes,
        planner_data=item.planner_data,
        storage_data=storage_data)


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

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # There's no storage input in the local plan
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:
        # Add a prefix for each rank
        # FIXME maybe make this the default behavior?
        for i, plan in enumerate(global_plan):
            plan.storage_data = _StoragePrefix(f"__{i}_")
        return global_plan

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
        res = []
        file_count = 0
        storage_plan: _StoragePrefix = plan.storage_data

        def gen_file(prefix: _StoragePrefix):
            nonlocal file_count
            file_name = f"{prefix.prefix}{file_count}"
            file_count += 1
            return file_name

        def _write_item(stream, write_item):
            data = planner.resolve_data(write_item)
            if write_item.is_bytesio:
                assert isinstance(data, io.BytesIO)
                stream.write(data.getbuffer())
            else:
                assert isinstance(data, torch.Tensor)
                torch.save(data, stream)

        # This is ugly, cleanup
        if self.single_file_per_rank:
            file_name = gen_file(storage_plan)
            with (self.path / file_name).open("wb") as w:
                for write_item in plan.items:
                    offset = w.tell()
                    _write_item(w, write_item)
                    length = w.tell() - offset
                    res.append(result_from_write_item(
                        write_item,
                        length,
                        _StorageInfo(file_name, offset, length)
                    ))
                os.fsync(w.fileno())

        else:
            for write_item in plan.items:
                file_name = gen_file(storage_plan)
                with (self.path / file_name).open("wb") as w:
                    _write_item(w, write_item)
                    length = w.tell()
                    os.fsync(w.fileno())
                    res.append(result_from_write_item(
                        write_item,
                        length,
                        _StorageInfo(file_name, 0, length)
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
            io.FileIO(file.fileno(), closefd=False),
            sinfo.offset, sinfo.length
        )

    def read_data(
        self,
        plan: LoadPlan,
        planner: LoadPlanner
    ) -> Future[None]:

        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            path = cast(_StorageInfo, read_item.storage_data).relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            with (self.path / relative_path).open("rb") as file:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    file_slice = self._slice_file(file, req.storage_data)

                    if req.is_bytesio:
                        planner.write_bytes(req, file_slice)
                    else:
                        tensor = cast(Tensor, torch.load(file_slice, map_location="cpu"))
                        tensor = tensor_narrow_n(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req)

                        assert (
                            target_tensor.size() == tensor.size()
                        ), f"req {req.fqn} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        target_tensor.copy_(tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with (self.path / ".metadata").open("rb") as metadata_file:
            md = pickle.load(metadata_file)
            return md

    def prepare_local_plan(self, metadata: Metadata, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan
