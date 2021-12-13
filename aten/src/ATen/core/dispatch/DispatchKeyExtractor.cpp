#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <sstream>

namespace c10 {

void DispatchKeyExtractor::setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough) {
  // (1) update nonFallthroughKeys_
  if (has_fallthrough) {
    nonFallthroughKeys_ = nonFallthroughKeys_.removeFunctionalityKey(k);
  } else {
    nonFallthroughKeys_ = nonFallthroughKeys_.add(k);
  }
  // (2) update nonFallthroughKeysPerBackend_
  if (k != DispatchKey::Undefined && (
      isPerBackendFunctionalityKey(k) ||
      isPerBackendFunctionalityKey(toFunctionalityKey(k)))) {
    // This is a per-backend functionality key.
    // We need to figure out what the currenty backend is,
    // and only update the bitset for that backend.
    auto backend_idx = static_cast<uint8_t>(toBackendKey(k));
    if (has_fallthrough) {
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].removeFunctionalityKey(k);
    } else {
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].add(k);
    }

    // Set requiresBitsetPerBackend_ accordingly
    for (size_t i = 0; i < static_cast<uint8_t>(DispatchKey::EndOfBackendKeys); ++i)   {
      if (nonFallthroughKeysPerBackend_[i] != nonFallthroughKeysPerBackend_[i+1]) {
        requiresBitsetPerBackend_ = true;
        return;
      }
    }
    requiresBitsetPerBackend_ = false;
    return;
  }
  // Otherwise, if a fallthrough is set for a functionality that isn't per backend,
  // Then we update the fallthrough bitset for EVERY backend.
  // TODO: we could probably optimize this by only lazily updating these values
  // the first time that we see requiresBitsetPerBackend_ = true
  // (which should almost never happen)
  for (size_t i = 0; i <= static_cast<uint8_t>(DispatchKey::EndOfBackendKeys); ++i) {
    if (has_fallthrough) {
      nonFallthroughKeysPerBackend_[i] = nonFallthroughKeysPerBackend_[i].removeFunctionalityKey(k);
    } else {
      nonFallthroughKeysPerBackend_[i] = nonFallthroughKeysPerBackend_[i].add(k);
    }
  }
}

std::string DispatchKeyExtractor::dumpState() const {
  std::ostringstream oss;
  for (size_t i=0; i < c10::utils::bitset::NUM_BITS(); ++i) {
    if (dispatch_arg_indices_reverse_.get(i)) {
      oss << "1";
    } else {
      oss << "0";
    }
  }
  oss << " " << nonFallthroughKeys_ << "\n";
  return oss.str();
}

void DispatchKeyExtractor::checkInvariants(const FunctionSchema& schema) const {
  TORCH_INTERNAL_ASSERT(makeBitsetForDispatchArgs(schema) == dispatch_arg_indices_reverse_);
}

} // namespace c10
