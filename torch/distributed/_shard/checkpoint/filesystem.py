from dataclasses import dataclass
import os
import dataclasses
import io
import pickle
from typing import List, Union, Dict, cast

import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path

from .metadata import (
    Metadata,
    MetadataIndex,
)
from .storage import (
    LoadItemType,
    LoadPlanner,
    LoadPlan,
    SavePlan,
    SavePlanner,
    StorageReader,
    StorageWriter,
    WriteItemType,
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
        index=item.index,
        size_in_bytes=size_in_bytes,
        storage_data=storage_data)

def _to_cpu_and_trim(tensor: Tensor) -> torch.Tensor:
    tensor = tensor.cpu()
    if tensor.storage().size() != tensor.numel():
        tensor = tensor.clone()
    return tensor

def _write_item(stream, data, write_item, file_name):
    offset = stream.tell()

    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        stream.write(data.getbuffer())
    else:
        assert isinstance(data, torch.Tensor)
        assert data.device == torch.device("cpu")
        torch.save(data, stream)
    length = stream.tell() - offset

    return result_from_write_item(
        write_item,
        length,
        _StorageInfo(file_name, offset, length)
    )

class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    N. B. All files are opened in exclusive mode, ranks do not override existing files and fail instead.
    """
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
        """
        super().__init__()
        self.path = Path(path)

    def init(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # There's no storage input in the local plan
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:
        self.path.mkdir(parents=True, exist_ok=True)

        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_")) for i, plan in enumerate(global_plan)
        ]
        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}"
            file_count += 1
            return file_name

        write_results = []
        for write_item in plan.items:
            file_name = gen_file()
            with open(self.path / file_name, "xb") as stream:
                data = planner.resolve_data(write_item)
                if write_item.type != WriteItemType.BYTE_IO:
                    tensor = cast(torch.Tensor, planner.resolve_data(write_item))
                    data = _to_cpu_and_trim(tensor)
                write_results.append(_write_item(stream, data, write_item, file_name))
                os.fsync(stream.fileno())

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(write_results)
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({
                wr.index: wr.storage_data for wr in wr_list
            })
        metadata.storage_data = storage_md
        with (self.path / ".metadata.tmp").open("xb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")

class FileSystemReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = Path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

    def read_data(
        self,
        plan: LoadPlan,
        planner: LoadPlanner
    ) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            with (self.path / relative_path).open("rb") as file:
                data: Union[io.BytesIO, torch.Tensor]
                if reqs[0].type == LoadItemType.BYTE_IO:
                    data = io.BytesIO(file.read(item_md.length))
                    data.seek(0)
                else:
                    data = cast(Tensor, torch.load(file, map_location="cpu"))

                for req in reqs:
                    if req.type == LoadItemType.BYTE_IO:
                        bytes = cast(io.BytesIO, data)
                        bytes.seek(0)
                        planner.load_bytes(req, bytes)
                    else:
                        tensor = cast(torch.Tensor, data)
                        tensor = tensor_narrow_n(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req)

                        assert (
                            target_tensor.size() == tensor.size()
                        ), f"req {req.index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        target_tensor.copy_(tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with (self.path / ".metadata").open("rb") as metadata_file:
            md = pickle.load(metadata_file)
            return md

    def init(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan
