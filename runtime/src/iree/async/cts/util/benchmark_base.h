// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base infrastructure for proactor benchmarks.
//
// Provides shared infrastructure for all proactor benchmarks, parallel to
// test_base.h for tests. Key components:
//
//   - ProactorCreateFn: Factory function type for creating proactors.
//   - NumaTopology: NUMA node discovery and current-thread placement.
//   - BenchmarkContext: Proactor lifetime and completion handling.
//   - Benchmark context helpers: Create/destroy, capability checking.
//   - Polling helpers: SpinPollUntilComplete for latency-sensitive benchmarks.
//
// Usage in benchmark files (e.g., cts/futex/futex_benchmark.cc):
//   #include "iree/async/cts/util/benchmark_base.h"
//   void RegisterFutexBenchmarks(const char* prefix, ProactorCreateFn fn);
//
// Usage in platform files (e.g., platform/io_uring/cts/futex_benchmark.cc):
//   #include "iree/async/cts/futex/futex_benchmark.h"
//   #include "iree/async/platform/io_uring/api.h"
//   RegisterFutexBenchmarks("io_uring", iree_async_proactor_create_io_uring);
//   BENCHMARK_MAIN();

#ifndef IREE_ASYNC_CTS_UTIL_BENCHMARK_BASE_H_
#define IREE_ASYNC_CTS_UTIL_BENCHMARK_BASE_H_

#include <atomic>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Proactor factory type
//===----------------------------------------------------------------------===//

// Factory function that creates a proactor backend.
// Matches the signature of iree_async_proactor_create_* functions.
typedef iree_status_t (*ProactorCreateFn)(iree_async_proactor_options_t options,
                                          iree_allocator_t allocator,
                                          iree_async_proactor_t** out_proactor);

//===----------------------------------------------------------------------===//
// NUMA topology
//===----------------------------------------------------------------------===//

// NUMA topology information discovered at benchmark startup.
struct NumaTopology {
  iree_host_size_t node_count = 1;
  iree_numa_node_id_t current_node = 0;
  bool is_multi_numa = false;

  // Discovers the NUMA topology for the current system.
  // Called once per benchmark context creation.
  static NumaTopology Discover() {
    NumaTopology topology;
    topology.node_count = iree_numa_node_count();
    topology.current_node = iree_numa_node_for_current_thread();
    topology.is_multi_numa = (topology.node_count > 1);
    return topology;
  }

  // Returns a different NUMA node than current_node, or IREE_NUMA_NODE_ANY
  // if there is only one node.
  iree_numa_node_id_t other_node() const {
    if (!is_multi_numa) return IREE_NUMA_NODE_ANY;
    // Pick a node that isn't the current one.
    for (iree_host_size_t i = 0; i < node_count; ++i) {
      if (static_cast<iree_numa_node_id_t>(i) != current_node) {
        return static_cast<iree_numa_node_id_t>(i);
      }
    }
    return current_node;  // Should not happen if is_multi_numa is true.
  }
};

//===----------------------------------------------------------------------===//
// Benchmark context
//===----------------------------------------------------------------------===//

// Default timeout for poll operations (5 seconds).
static constexpr iree_duration_t kDefaultPollBudget = 5000 * 1000000LL;

// Base context for CTS benchmarks.
// Manages proactor lifetime and provides common helpers.
struct BenchmarkContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_proactor_capabilities_t capabilities = 0;
  NumaTopology numa;
  std::atomic<int> completions{0};
  iree_status_code_t last_status_code = IREE_STATUS_OK;

  // Completion callback that increments the completion counter.
  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* context = static_cast<BenchmarkContext*>(user_data);
    context->last_status_code = iree_status_code(status);
    context->completions.fetch_add(1, std::memory_order_release);
    iree_status_ignore(status);
  }

  void Reset() {
    completions.store(0, std::memory_order_release);
    last_status_code = IREE_STATUS_OK;
  }

  // Blocking poll until expected completions are received.
  bool PollUntilComplete(int expected,
                         iree_duration_t budget_ns = kDefaultPollBudget) {
    iree_time_t deadline = iree_time_now() + budget_ns;
    while (completions.load(std::memory_order_acquire) < expected) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t count = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor, iree_make_timeout_ms(100), &count);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        return false;
      }
      iree_status_ignore(status);
    }
    return last_status_code == IREE_STATUS_OK;
  }

  // Spin poll with immediate timeout for latency-sensitive benchmarks.
  // Does NOT yield - use this only inside manually-timed regions where
  // accuracy matters. For longer waits, use PollUntilComplete().
  bool SpinPollUntilComplete(int expected,
                             iree_duration_t budget_ns = kDefaultPollBudget) {
    iree_time_t deadline = iree_time_now() + budget_ns;
    while (completions.load(std::memory_order_acquire) < expected) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t count = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor, iree_immediate_timeout(), &count);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        return false;
      }
      iree_status_ignore(status);
    }
    return last_status_code == IREE_STATUS_OK;
  }
};

// Creates a benchmark context with a proactor from the given factory.
// Returns nullptr and calls state.SkipWithError() on failure.
// This is the preferred overload for link-time composed benchmarks.
inline BenchmarkContext* CreateBenchmarkContext(const ProactorFactory& factory,
                                                ::benchmark::State& state) {
  auto* context = new BenchmarkContext();

  auto result = factory();
  if (!result.ok()) {
    if (result.status().code() == iree::StatusCode::kUnavailable) {
      state.SkipWithError("Backend unavailable on this system");
    } else {
      state.SkipWithError("Proactor creation failed");
    }
    delete context;
    return nullptr;
  }
  context->proactor = result.value();

  context->capabilities =
      iree_async_proactor_query_capabilities(context->proactor);
  context->numa = NumaTopology::Discover();

  return context;
}

// Legacy overload for backwards compatibility with ProactorCreateFn.
// Prefer using the ProactorFactory overload for new code.
inline BenchmarkContext* CreateBenchmarkContext(ProactorCreateFn create_fn,
                                                ::benchmark::State& state) {
  auto* context = new BenchmarkContext();

  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  iree_status_t status =
      create_fn(options, iree_allocator_system(), &context->proactor);
  if (!iree_status_is_ok(status)) {
    if (iree_status_is_unavailable(status)) {
      state.SkipWithError("Backend unavailable on this system");
    } else {
      state.SkipWithError("Proactor creation failed");
    }
    iree_status_ignore(status);
    delete context;
    return nullptr;
  }

  context->capabilities =
      iree_async_proactor_query_capabilities(context->proactor);
  context->numa = NumaTopology::Discover();

  return context;
}

// Destroys a benchmark context created with CreateBenchmarkContext().
inline void DestroyBenchmarkContext(BenchmarkContext* context) {
  if (!context) return;
  iree_async_proactor_release(context->proactor);
  delete context;
}

// Skips the benchmark if the given capability is missing.
// Returns false if skipped, true if the capability is present.
inline bool RequireCapability(BenchmarkContext* context,
                              iree_async_proactor_capabilities_t required,
                              ::benchmark::State& state) {
  if (!(context->capabilities & required)) {
    state.SkipWithError("Backend lacks required capability");
    return false;
  }
  return true;
}

// Skips the benchmark if the system is not multi-NUMA.
// Returns false if skipped, true if multi-NUMA is available.
inline bool RequireMultiNuma(BenchmarkContext* context,
                             ::benchmark::State& state) {
  if (!context->numa.is_multi_numa) {
    state.SkipWithError("Single NUMA node system");
    return false;
  }
  return true;
}

}  // namespace iree::async::cts

#endif  // IREE_ASYNC_CTS_UTIL_BENCHMARK_BASE_H_
