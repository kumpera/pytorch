#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

Backend::Backend(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.backend");
}

Backend::~Backend() = default;

void Backend::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.backend_{}", getBackendName()));
}

void Backend::registerForDebug(
    c10::intrusive_ptr<::c10d::Store> store,
    const std::string pg_name) {
  debug_store_ = store;
  debug_pg_name_ = pg_name;

  // FIXME this is a prototype so pretend usage is ok

  std::stringstream ss;
  ss << pg_name << "$" << size_ << "$" << rank_;
  auto x = ss.str();
  debug_store_->set("register", ss.str());
  printf("registering %s\n", x.c_str());
  std::vector<std::string> keys = {pg_name + "$ready"};

  // This is a store barrier
  debug_store_->wait(keys);
}

void Backend::emitEvent(
    const std::string& event,
    const std::vector<uint8_t>& data) {

  //XXX taking a std::vector<uint8_t> sucks, we should take a span, but we're not c++ 20 yet
  std::stringstream ss;
  ss << debug_pg_name_ << "$" << rank_ << "$" << event;
  auto x = ss.str();
  printf("emit event %s %s\n", x.c_str(), data.data());

  debug_store_->set(ss.str(), data);
}

} // namespace c10d
