// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/signal_pool.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

namespace {

//===----------------------------------------------------------------------===//
// Shared HSA context (one per process, like gtest's SetUpTestSuite)
//===----------------------------------------------------------------------===//

class SignalPoolBenchmark : public benchmark::Fixture {
 public:
  static void InitializeOnce() {
    if (initialized_) return;
    initialized_ = true;
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return;
    }
    status =
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa_, &topology_);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
      return;
    }
    if (topology_.gpu_agent_count == 0) {
      iree_hal_amdgpu_topology_deinitialize(&topology_);
      iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
      return;
    }
    available_ = true;
  }

  static void DeinitializeOnce() {
    if (!available_) return;
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
    available_ = false;
  }

  void SetUp(benchmark::State& state) override {
    InitializeOnce();
    if (!available_) {
      state.SkipWithError("HSA not available or no GPU devices");
    }
  }

 protected:
  static bool initialized_;
  static bool available_;
  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

bool SignalPoolBenchmark::initialized_ = false;
bool SignalPoolBenchmark::available_ = false;
iree_allocator_t SignalPoolBenchmark::host_allocator_;
iree_hal_amdgpu_libhsa_t SignalPoolBenchmark::libhsa_;
iree_hal_amdgpu_topology_t SignalPoolBenchmark::topology_;

//===----------------------------------------------------------------------===//
// Baseline: raw hsa_amd_signal_create + hsa_signal_destroy
//===----------------------------------------------------------------------===//
// Measures ROCR's SharedSignalPool pop+push plus signal construction.
// This includes ROCR's process-global HybridMutex.

BENCHMARK_DEFINE_F(SignalPoolBenchmark, RawHsaSignalCreateDestroy)
(benchmark::State& state) {
  for (auto _ : state) {
    hsa_signal_t signal = {0};
    IREE_CHECK_OK(iree_hsa_amd_signal_create(
        IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
        /*consumers=*/NULL, /*attributes=*/0, &signal));
    benchmark::DoNotOptimize(signal);
    IREE_CHECK_OK(iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), signal));
  }
}
BENCHMARK_REGISTER_F(SignalPoolBenchmark, RawHsaSignalCreateDestroy);

//===----------------------------------------------------------------------===//
// Host signal pool: acquire + release (steady state)
//===----------------------------------------------------------------------===//
// Measures our pool's LIFO pop+push plus hsa_signal_store_relaxed for value
// reset. The HSA signal memory lives in CPU kernarg memory (placed by ROCR),
// so the relaxed store is a local system RAM write with no PCIe traffic.

BENCHMARK_DEFINE_F(SignalPoolBenchmark, HostPoolAcquireRelease)
(benchmark::State& state) {
  iree_hal_amdgpu_host_signal_pool_t pool;
  IREE_CHECK_OK(iree_hal_amdgpu_host_signal_pool_initialize(
      &libhsa_, /*initial_capacity=*/64, /*batch_size=*/32, host_allocator_,
      &pool));

  for (auto _ : state) {
    hsa_signal_t signal = {0};
    IREE_CHECK_OK(iree_hal_amdgpu_host_signal_pool_acquire(&pool, 1, &signal));
    benchmark::DoNotOptimize(signal);
    iree_hal_amdgpu_host_signal_pool_release(&pool, signal);
  }

  iree_hal_amdgpu_host_signal_pool_deinitialize(&pool);
}
BENCHMARK_REGISTER_F(SignalPoolBenchmark, HostPoolAcquireRelease);

}  // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  SignalPoolBenchmark::DeinitializeOnce();
  return 0;
}
