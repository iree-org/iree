// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/kfd.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// NOTE: ROCR also opens the KFD - if it initializes then we're likely to
// succeed as well. We need information we can only get from HSA to make the
// ioctls so we have to setup a full topology here.
struct KFDTest : public ::testing::Test {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_hal_amdgpu_topology_t topology;

  void SetUp() override {
    IREE_TRACE_SCOPE();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  std::optional<uint32_t> GetFirstGPUAgentUID() {
    uint32_t gpu_uid = 0;
    iree_status_t status = iree_hsa_agent_get_info(
        IREE_LIBHSA(&libhsa), topology.gpu_agents[0],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID, &gpu_uid);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return std::nullopt;
    }
    return gpu_uid;
  }
};

// Tests opening and closing the KFD. It should not crash/leak/etc.
// Note that we currently no-op the helpers on non-Linux platforms and this
// will always succeed.
TEST_F(KFDTest, Lifetime) {
  int kfd = -1;
  IREE_ASSERT_OK(iree_hal_amdgpu_kfd_open(&kfd));
  iree_hal_amdgpu_kfd_close(kfd);
}

// Tests that we get non-zero counters from the clock.
// We always make the call but only expect non-zero if the returned kfd fd is
// not 0 (the special value for "not a real fd" we use on non-Linux).
TEST_F(KFDTest, GetClockCounters) {
  // Find a GPU ID we can use to make the ioctl. If we can't find one we skip
  // the test.
  auto gpu_uid = GetFirstGPUAgentUID();
  if (!gpu_uid) {
    GTEST_SKIP() << "no GPU agent UID available";
    return;
  }

  int kfd = -1;
  IREE_ASSERT_OK(iree_hal_amdgpu_kfd_open(&kfd));

  iree_hal_amdgpu_clock_counters_t counters = {0};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_kfd_get_clock_counters(kfd, *gpu_uid, &counters));

  if (kfd != 0) {
    // Don't care about the values, just that they were populated.
    ASSERT_NE(counters.gpu_clock_counter, 0);
    ASSERT_NE(counters.cpu_clock_counter, 0);
    ASSERT_NE(counters.system_clock_counter, 0);
    ASSERT_NE(counters.system_clock_freq, 0);
  }

  iree_hal_amdgpu_kfd_close(kfd);
}

}  // namespace
}  // namespace iree::hal::amdgpu
