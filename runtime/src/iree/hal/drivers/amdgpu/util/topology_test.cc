// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/topology.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

struct TopologyTest : public ::testing::Test {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_amdgpu_libhsa_t libhsa;

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
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};

TEST_F(TopologyTest, Empty) {
  iree_hal_amdgpu_topology_t topology;
  iree_hal_amdgpu_topology_initialize(&topology);
  // Need at least 1 CPU and GPU agent.
  EXPECT_THAT(Status(iree_hal_amdgpu_topology_verify(&topology, &libhsa)),
              StatusIs(StatusCode::kInvalidArgument));
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

// Tests that we get at least one CPU and GPU agent.
// If we don't get any but still succeed it means there are none available and
// we can skip the test.
TEST_F(TopologyTest, InitializeWithDefaults) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
  if (topology.gpu_agent_count == 0) {
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_GE(topology.all_agent_count, 2);
  EXPECT_GE(topology.cpu_agent_count, 1);
  EXPECT_GE(topology.gpu_agent_count, 1);
  EXPECT_GE(topology.gpu_agent_queue_count, 1);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

// Tests that initialize_from_path with no path provided is the same as
// initialize defaults.
TEST_F(TopologyTest, InitializeFromPathEmpty) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_path(
      &libhsa, IREE_SV(""), &topology));
  if (topology.gpu_agent_count == 0) {
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_GE(topology.all_agent_count, 2);
  EXPECT_GE(topology.cpu_agent_count, 1);
  EXPECT_GE(topology.gpu_agent_count, 1);
  EXPECT_GE(topology.gpu_agent_queue_count, 1);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

// Tests initialization from a string ordinal. There should always be at least 1
// agent unless the user has set ROCR_VISIBLE_DEVICES to nothing (which may not
// even be valid).
TEST_F(TopologyTest, InitializeFromPathOrdinal) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_path(
      &libhsa, IREE_SV("0"), &topology));
  if (topology.gpu_agent_count == 0) {
    // This could be ignoring an error, but it usually just indicates no agents
    // on the machine.
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_EQ(topology.all_agent_count, 2);
  EXPECT_EQ(topology.cpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_queue_count, 1);
  EXPECT_EQ(topology.gpu_cpu_map[0], 0);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

// Tests that initialize_from_gpu_agent_mask with a 0 mask is the same as
// initializing from defaults.
TEST_F(TopologyTest, InitializeFromGPUAgentMask0) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(
      &libhsa, 0ull, &topology));
  if (topology.gpu_agent_count == 0) {
    // This could be ignoring an error, but it usually just indicates no agents
    // on the machine.
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_GE(topology.all_agent_count, 2);
  EXPECT_GE(topology.cpu_agent_count, 1);
  EXPECT_GE(topology.gpu_agent_count, 1);
  EXPECT_GE(topology.gpu_agent_queue_count, 1);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

// Tests initializing a single device with the given ordinal mask.
TEST_F(TopologyTest, InitializeFromGPUAgentMask1) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(
      &libhsa, 1ull << 0, &topology));
  if (topology.gpu_agent_count == 0) {
    // This could be ignoring an error, but it usually just indicates no agents
    // on the machine.
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_EQ(topology.all_agent_count, 2);
  EXPECT_EQ(topology.cpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_queue_count, 1);
  EXPECT_EQ(topology.gpu_cpu_map[0], 0);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

}  // namespace
}  // namespace iree::hal::amdgpu
