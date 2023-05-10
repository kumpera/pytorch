#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>

namespace c10d {

ProcessGroup::BackendType strToBackendType(std::string backend) {
  if (backend == "undefined") {
    return ProcessGroup::BackendType::UNDEFINED;
  } else if (backend == "gloo") {
    return ProcessGroup::BackendType::GLOO;
  } else if (backend == "nccl") {
    return ProcessGroup::BackendType::NCCL;
  } else if (backend == "ucc") {
    return ProcessGroup::BackendType::UCC;
  } else if (backend == "mpi") {
    return ProcessGroup::BackendType::MPI;
  } else {
    return ProcessGroup::BackendType::CUSTOM;
  }
}

std::string backendTypeToStr(ProcessGroup::BackendType backendType) {
  switch (backendType) {
    case ProcessGroup::BackendType::UNDEFINED:
      return "undefined";
    case ProcessGroup::BackendType::GLOO:
      return "gloo";
    case ProcessGroup::BackendType::NCCL:
      return "nccl";
    case ProcessGroup::BackendType::UCC:
      return "ucc";
    case ProcessGroup::BackendType::MPI:
      return "mpi";
    case ProcessGroup::BackendType::CUSTOM:
      return "custom";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown backend type");
  }
}

std::string opTypeToString(OpType opType) {
  switch (opType) {
    case OpType::BROADCAST:
      return "BROADCAST";
    case OpType::ALLREDUCE:
      return "ALLREDUCE";
    case OpType::ALLREDUCE_COALESCED:
      return "ALLREDUCE_COALESCED";
    case OpType::REDUCE:
      return "REDUCE";
    case OpType::ALLGATHER:
      return "ALLGATHER";
    case OpType::_ALLGATHER_BASE:
      return "_ALLGATHER_BASE";
    case OpType::ALLGATHER_COALESCED:
      return "ALLGATHER_COALESCED";
    case OpType::GATHER:
      return "GATHER";
    case OpType::SCATTER:
      return "SCATTER";
    case OpType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case OpType::ALLTOALL_BASE:
      return "ALLTOALL_BASE";
    case OpType::ALLTOALL:
      return "ALLTOALL";
    case OpType::SEND:
      return "SEND";
    case OpType::RECV:
      return "RECV";
    case OpType::RECVANYSOURCE:
      return "RECVANYSOURCE";
    case OpType::BARRIER:
      return "BARRIER";
    case OpType::UNKNOWN:
      return "UNKNOWN";
    case OpType::_REDUCE_SCATTER_BASE:
      return "_REDUCE_SCATTER_BASE";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown op type!");
  }
  return "UNKNOWN";
}

bool isP2POp(OpType opType, bool batchP2P /*= false*/) {
  if (batchP2P)
    return false;
  return opType == OpType::SEND || opType == OpType::RECV ||
      opType == OpType::RECVANYSOURCE;
}

c10::intrusive_ptr<Backend> ProcessGroup::getBackend(
    c10::DeviceType deviceType) {
  // If there is a backend associated with this device type then return it
  if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
    return deviceTypeToBackend_.at(deviceType);
  }

  // Get the backend type associated with the device
  ProcessGroup::BackendType backendType;
  try {
    backendType = deviceTypeToBackendType_.at(deviceType);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(
        false, "No backend type associated with device type ", deviceType);
  }

  // Check if the backend has already been initialized
  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto backend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = backend;
    return backend;
  }

  TORCH_CHECK(
      false,
      "Could not retrieve or create the backend ",
      backendType,
      " for device type ",
      deviceType);
}

ProcessGroup::ProcessGroup(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : store_(store),
      rank_(rank),
      size_(size),
      options_(options),
      backendType_(strToBackendType(options->backend)),
      dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), backendType_(BackendType::UNDEFINED) {}

ProcessGroup::~ProcessGroup() = default;

void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(
      fmt::format("c10d.process_group_{}", getBackendName()));
}

std::vector<c10::intrusive_ptr<Work>> ProcessGroup::batchExecute(
    const c10::intrusive_ptr<CollectivesBatch>& batch,
    std::vector<at::Tensor>& tensors) {
  // TODO do some validation prior to starting
  // TODO timeout

  int param_idx = 0;
  std::vector<c10::intrusive_ptr<Work>> result;
  result.reserve(batch->collectives.size());
  std::vector<at::Tensor> input_tensors;

  for (const auto i : c10::irange(batch->collectives.size())) {
    switch (batch->collectives[i]) {
      case CollType::ALL_REDUCE: {
        input_tensors.resize(1);
        input_tensors[0] = tensors[param_idx];

        AllreduceOptions opts{};
        opts.reduceOp = batch->reduceOps[i];

        result.emplace_back(this->allreduce(input_tensors, opts));
        ++param_idx;
        break;
      }
      case CollType::ALL_GATHER: {
        AllgatherOptions opts{};
        result.emplace_back(this->_allgather_base(
            tensors[param_idx], tensors[param_idx + 1], opts));
        param_idx += 2;
        break;
      }
      case CollType::REDUCE_SCATTER: {
        ReduceScatterOptions opts{};
        result.emplace_back(this->_reduce_scatter_base(
            tensors[param_idx], tensors[param_idx + 1], opts));
        param_idx += 2;
        break;
      }
    }
  }
  return result;
}

} // namespace c10d
