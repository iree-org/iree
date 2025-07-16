// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/system.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct SystemTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
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

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t SystemTest::host_allocator;
iree_hal_amdgpu_libhsa_t SystemTest::libhsa;
iree_hal_amdgpu_topology_t SystemTest::topology;

TEST_F(SystemTest, Lifetime) {
  IREE_TRACE_SCOPE();

  iree_hal_amdgpu_system_options_t options = {0};
  options.exclusive_execution = 0;
  options.trace_execution = 0;

  iree_hal_amdgpu_system_t* system = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_system_allocate(&libhsa, &topology, options,
                                                 host_allocator, &system));

  // libhsa should have been cloned into the system. This just checks that the
  // pointers match (non-NULL).
  EXPECT_EQ(system->libhsa.hsa_agent_get_info, libhsa.hsa_agent_get_info);

  // Topology should have been cloned into the system.
  // This simply checks that the same agents exist and their ordering is the
  // same (as ordinals must be consistent).
  ASSERT_EQ(system->topology.cpu_agent_count, topology.cpu_agent_count);
  for (iree_host_size_t i = 0; i < topology.cpu_agent_count; ++i) {
    EXPECT_EQ(system->topology.cpu_agents[i].handle,
              topology.cpu_agents[i].handle);
  }
  ASSERT_EQ(system->topology.gpu_agent_count, topology.gpu_agent_count);
  for (iree_host_size_t i = 0; i < topology.gpu_agent_count; ++i) {
    EXPECT_EQ(system->topology.gpu_agents[i].handle,
              topology.gpu_agents[i].handle);
  }

  // System info should have valid values. They'll vary per system so we can't
  // check that they match anything but we do know the timestamp frequency will
  // always != 0.
  EXPECT_NE(system->info.timestamp_frequency, 0);

  // Device library gets loaded/initialized for the topology.
  // Assert that it is referencing the system libhsa clone instead of the
  // original one passed in from the test state.
  EXPECT_EQ(system->device_library.libhsa, &system->libhsa);

  // Memory pools for each CPU agent must be present.
  for (iree_host_size_t i = 0; i < topology.cpu_agent_count; ++i) {
    EXPECT_NE(system->host_memory_pools[i].fine_region.handle, 0);
    EXPECT_NE(system->host_memory_pools[i].fine_pool.handle, 0);
  }

  iree_hal_amdgpu_system_free(system);
}

}  // namespace
}  // namespace iree::hal::amdgpu
