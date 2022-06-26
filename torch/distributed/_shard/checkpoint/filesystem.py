import collections
import math
from operator import itemgetter
from dataclasses import dataclass
import os
import dataclasses
import io
import pickle
from typing import List, Union, Dict, cast

import torch
from torch import Tensor
from torch.backends import cuda
from torch.cuda import current_stream
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


def _tensor_size(tensor):
    return tensor.numel() * tensor.element_size()


class _SerialCpuLoader:
    def __init__(self, resolve_fun):
        self.resolve_fun = resolve_fun
        self.items = []
    
    def add(self, size, obj):
        self.items.append((size, obj))

    def values(self):
        for size, obj in self.items:
            tensor = self.resolve_fun(obj)
            tensor = tensor.cpu()
            yield (tensor, obj,)

class _OverlappingCpuLoader:
    def __init__(self, resolve_fun, stream = None, inflight_threshhold = 1_000_000):
        self.resolve_fun = resolve_fun
        self.items = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items = collections.deque()
        self.idx = 0
        self.stream = stream or torch.cuda.current_stream()

    def add(self, size, obj):
        self.items.append((size, obj))

    @property
    def _done(self):
        return self.idx >= len(self.items)

    def _drain(self):
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self):
        with torch.cuda.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.is_cuda:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                elif tensor.device == torch.device("cpu"):
                    if tensor.storage().size() != tensor.numel():
                        # this forces the tensor to be both contiguous and with minimal storage
                        tensor = tensor.clone()

                self.current_items.append((tensor, obj,))
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self):
        assert self._done
        self.stream.synchronize()
        return self.current_items

    def values(self):
        self.items.sort(key=lambda x: x[0])
        while not self._done:
            drained = self._drain()
            self._refill()
            for obj in drained:
                yield obj

        for val in self._finish():
            yield val

class _ParallelOverlappingCpuLoader:
    def __init__(self, resolve_fun, inflight_threshhold = 1_000_000, stream_count = 4):
        self.resolve_fun = resolve_fun
        self.inflight_threshhold = inflight_threshhold
        self.stream_count = stream_count
        self.items = []
        self.loaders = []
    
    def add(self, size, obj):
        self.items.append((size, obj))

    @property
    def _done(self):
        return all(l._done for l in self.loaders)

    def values(self):
        # must sync before we splinter into multiple streams
        # FIXME add a dependency across steams instead
        torch.cuda.synchronize()

        buckets = [[] for _ in range(self.stream_count)]
        bucket_sizes = [0 for _ in range(self.stream_count)]
        streams = []
        for size, obj in self.items:
            min_idx = min(enumerate(bucket_sizes), key=itemgetter(1))[0]
            buckets[min_idx].append((size,obj,))
            bucket_sizes[min_idx] += size

        for i, bucket in enumerate(buckets):
            stream = torch.cuda.current_stream() if i == 0 else torch.cuda.Stream()
            streams.append(stream)
            l = _OverlappingCpuLoader(
                self.resolve_fun,
                stream=stream,
                inflight_threshhold=self.inflight_threshhold)
            l.items.extend(bucket)
            self.loaders.append(l)

        while not self._done:
            drained = []
            for l in self.loaders:
                drained.extend(l._drain())
            for l in self.loaders:
                l._refill()

            for obj in drained:
                yield obj

        for l in self.loaders:
            for val in l._finish():
                yield val
            
class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    def __init__(
        self, 
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = False,
        sync_files = True
    ) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
        """
        super().__init__()
        self.path = Path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files

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
        res = []
        file_count = 0
        storage_plan: _StoragePrefix = plan.storage_data

        def gen_file(prefix: _StoragePrefix):
            nonlocal file_count
            file_name = f"{prefix.prefix}{file_count}"
            file_count += 1
            return file_name

        def _write_item(stream, data, write_item, storage_key, write_results):
            offset = stream.tell()
            
            if write_item.type == WriteItemType.BYTE_IO:
                assert isinstance(data, io.BytesIO)
                stream.write(data.getbuffer())
            else:
                assert isinstance(data, torch.Tensor)
                assert data.device == torch.device("cpu")
                torch.save(data, stream)
            length = stream.tell() - offset

            write_results.append(result_from_write_item(
                write_item,
                length,
                _StorageInfo(storage_key, offset, length)
            ))

        # This is ugly, cleanup -
        # I have an idea, we pair (file, [items]) and go to town
        if self.single_file_per_rank:
            file_name = gen_file(storage_plan)
            with (self.path / file_name).open("wb") as w:
                bytes_w = [wi for wi in plan.items if wi.type == WriteItemType.BYTE_IO]
                tensor_w = [wi for wi in plan.items if wi.type != WriteItemType.BYTE_IO]

                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    _write_item(w, data, write_item, file_name, res)

                # loader = _ParallelOverlappingCpuLoader(
                #     lambda x: planner.resolve_data(x),
                #     inflight_threshhold=10_000_000,
                #     stream_count=6)
                loader = _OverlappingCpuLoader(
                    lambda x: planner.resolve_data(x),
                    inflight_threshhold=10_000_000
                )
                for write_item in tensor_w:
                    #fixme multiply by element size for better LB
                    tensor_size = math.prod(write_item.tensor_data.info.size)
                    loader.add(tensor_size, write_item)

                for tensor, write_item in loader.values():
                    _write_item(w, tensor, write_item, file_name, res)
                if self.sync_files:
                    os.fsync(w.fileno())

        else:
            for write_item in plan.items:
                file_name = gen_file(storage_plan)
                with (self.path / file_name).open("wb") as w:
                    data = planner.resolve_data(write_item)
                    if isinstance(data, torch.Tensor):
                        data = data.cpu()
                    _write_item(w, data, write_item, file_name, res)
                    if self.sync_files:
                        os.fsync(w.fileno())

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(res)
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({
                wr.index: wr.storage_data for wr in wr_list
            })
        metadata.storage_data = storage_md
        with (self.path / ".metadata.tmp").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")


class SlicedBufferedReader(io.BufferedReader):
    # TODO override read to handle (-1) correctly
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
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

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
            item_md = self.storage_data[read_item.index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            with (self.path / relative_path).open("rb") as file:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.index]
                    file_slice = self._slice_file(file, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        bytes = io.BytesIO(file_slice.read(item_md.length))
                        bytes.seek(0)
                        planner.load_bytes(req, bytes)
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

    def init(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan
