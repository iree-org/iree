// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct DeviceLibraryTest : public ::testing::Test {
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
};

TEST_F(DeviceLibraryTest, Load) {
  IREE_TRACE_SCOPE();

  iree_hal_amdgpu_device_library_t library;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_library_initialize(
      &libhsa, &topology, host_allocator, &library));

  for (iree_host_size_t i = 0; i < topology.gpu_agent_count; ++i) {
    hsa_agent_t agent = topology.gpu_agents[i];

    // Code range should be non-zero and somewhere in the process address space.
    iree_hal_amdgpu_code_range_t code_range = {0};
    IREE_ASSERT_OK(iree_hal_amdgpu_device_library_populate_agent_code_range(
        &library, agent, &code_range));
    ASSERT_NE(code_range.host_ptr, nullptr);
    ASSERT_NE(code_range.device_ptr, 0);
    ASSERT_NE(code_range.size, 0);

    // Kernel information should be populated but we don't really care with what
    // here as we aren't running anything.
    iree_hal_amdgpu_device_kernels_t kernels;
    IREE_ASSERT_OK(iree_hal_amdgpu_device_library_populate_agent_kernels(
        &library, agent, &kernels));
    const auto& any_args = kernels.iree_hal_amdgpu_device_buffer_fill_x1;
    ASSERT_NE(any_args.kernel_object, 0);
    ASSERT_NE(any_args.setup, 0);
    ASSERT_NE(any_args.kernarg_size, 0);
    ASSERT_NE(any_args.kernarg_alignment, 0);
  }

  iree_hal_amdgpu_device_library_deinitialize(&library);
}

}  // namespace
}  // namespace iree::hal::amdgpu
