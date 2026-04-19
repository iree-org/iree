// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/topology.h"

#include <string>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static hsa_agent_t MakeFakeAgent(uint64_t handle) {
  hsa_agent_t agent;
  agent.handle = handle;
  return agent;
}

static const iree_hal_amdgpu_libhsa_t* FakeLibHsa() {
  static const iree_hal_amdgpu_libhsa_t libhsa = {};
  return &libhsa;
}

static iree_hal_amdgpu_topology_t MakeStructurallyValidTopology() {
  iree_hal_amdgpu_topology_t topology;
  iree_hal_amdgpu_topology_initialize(&topology);
  topology.gpu_agent_queue_count = 1;
  topology.cpu_agent_count = 1;
  topology.cpu_agents[0] = MakeFakeAgent(1);
  topology.gpu_agent_count = 1;
  topology.gpu_agents[0] = MakeFakeAgent(2);
  topology.all_agent_count = 2;
  topology.all_agents[0] = topology.cpu_agents[0];
  topology.all_agents[1] = topology.gpu_agents[0];
  topology.gpu_cpu_map[0] = 0;
  return topology;
}

static void ExpectTopologyHasTwoGpus(
    const iree_hal_amdgpu_topology_t& topology) {
  EXPECT_GE(topology.all_agent_count, 3);
  EXPECT_GE(topology.cpu_agent_count, 1);
  ASSERT_EQ(topology.gpu_agent_count, 2);
  EXPECT_GE(topology.gpu_agent_queue_count, 1);
  for (iree_host_size_t i = 0; i < topology.gpu_agent_count; ++i) {
    EXPECT_LT(topology.gpu_cpu_map[i], topology.cpu_agent_count);
  }
}

static iree_status_t AppendAgentUuidPathFragment(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    std::string* path) {
  char agent_uuid[32] = {0};
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID,
      agent_uuid));
  if (!path->empty()) path->append(",");
  path->append(agent_uuid);
  return iree_ok_status();
}

TEST(TopologyStructureTest, VerifyAcceptsStructurallyValidTopology) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  IREE_EXPECT_OK(iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

TEST(TopologyStructureTest, VerifyRejectsStorageCountBeyondCapacity) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  topology.cpu_agent_count = IREE_HAL_AMDGPU_MAX_CPU_AGENT + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

TEST(TopologyStructureTest, VerifyRejectsAllAgentCountMismatch) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  topology.all_agent_count = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

TEST(TopologyStructureTest, VerifyRejectsQueueSpaceOverflow) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  topology.gpu_agent_queue_count = IREE_HAL_MAX_QUEUES + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

TEST(TopologyStructureTest, VerifyRejectsGpuCpuMapOutOfRange) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  topology.gpu_cpu_map[0] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

TEST(TopologyStructureTest, VerifyRejectsDuplicateAllAgents) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  topology.all_agents[1] = topology.all_agents[0];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

TEST(TopologyStructureTest, VerifyRejectsGpuMissingFromAllAgents) {
  iree_hal_amdgpu_topology_t topology = MakeStructurallyValidTopology();
  topology.all_agents[1] = MakeFakeAgent(3);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_topology_verify(&topology, FakeLibHsa()));
}

struct TopologyTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t TopologyTest::host_allocator;
iree_hal_amdgpu_libhsa_t TopologyTest::libhsa;

TEST_F(TopologyTest, Empty) {
  iree_hal_amdgpu_topology_t topology;
  iree_hal_amdgpu_topology_initialize(&topology);
  // Need at least 1 CPU and GPU agent.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_topology_verify(&topology, &libhsa));
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

TEST_F(TopologyTest, InsertCpuAgentAllowsLastSlot) {
  iree_hal_amdgpu_topology_t defaults;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &defaults));
  if (defaults.cpu_agent_count == 0) {
    iree_hal_amdgpu_topology_deinitialize(&defaults);
    GTEST_SKIP() << "no CPU agents found";
    return;
  }
  hsa_agent_t cpu_agent = defaults.cpu_agents[0];
  iree_hal_amdgpu_topology_deinitialize(&defaults);

  iree_hal_amdgpu_topology_t topology;
  iree_hal_amdgpu_topology_initialize(&topology);
  topology.cpu_agent_count = IREE_HAL_AMDGPU_MAX_CPU_AGENT - 1;
  topology.all_agent_count = topology.cpu_agent_count;
  for (iree_host_size_t i = 0; i < topology.cpu_agent_count; ++i) {
    hsa_agent_t fake_agent = MakeFakeAgent(0xCAFE000000000000ull + i);
    topology.cpu_agents[i] = fake_agent;
    topology.all_agents[i] = fake_agent;
  }

  iree_host_size_t index = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_insert_cpu_agent(&topology, &libhsa,
                                                           cpu_agent, &index));
  EXPECT_EQ(index, IREE_HAL_AMDGPU_MAX_CPU_AGENT - 1);
  EXPECT_EQ(topology.cpu_agent_count, IREE_HAL_AMDGPU_MAX_CPU_AGENT);
  EXPECT_EQ(topology.all_agent_count, IREE_HAL_AMDGPU_MAX_CPU_AGENT);
  EXPECT_EQ(topology.cpu_agents[index].handle, cpu_agent.handle);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

TEST_F(TopologyTest, InsertGpuAgentAllowsLastSlot) {
  iree_hal_amdgpu_topology_t defaults;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &defaults));
  if (defaults.gpu_agent_count == 0 || defaults.cpu_agent_count == 0) {
    iree_hal_amdgpu_topology_deinitialize(&defaults);
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  hsa_agent_t cpu_agent = defaults.cpu_agents[0];
  hsa_agent_t gpu_agent = defaults.gpu_agents[0];
  iree_hal_amdgpu_topology_deinitialize(&defaults);

  iree_hal_amdgpu_topology_t topology;
  iree_hal_amdgpu_topology_initialize(&topology);
  topology.cpu_agent_count = 1;
  topology.cpu_agents[0] = cpu_agent;
  topology.all_agent_count = 1;
  topology.all_agents[0] = cpu_agent;
  topology.gpu_agent_count = IREE_HAL_AMDGPU_MAX_GPU_AGENT - 1;
  for (iree_host_size_t i = 0; i < topology.gpu_agent_count; ++i) {
    hsa_agent_t fake_agent = MakeFakeAgent(0xC0DE000000000000ull + i);
    topology.gpu_agents[i] = fake_agent;
    topology.gpu_cpu_map[i] = 0;
    topology.all_agents[topology.all_agent_count++] = fake_agent;
  }

  IREE_ASSERT_OK(iree_hal_amdgpu_topology_insert_gpu_agent(
      &topology, &libhsa, gpu_agent, cpu_agent));
  EXPECT_EQ(topology.gpu_agent_count, IREE_HAL_AMDGPU_MAX_GPU_AGENT);
  EXPECT_EQ(topology.all_agent_count, IREE_HAL_AMDGPU_MAX_GPU_AGENT + 1);
  EXPECT_EQ(topology.gpu_agents[IREE_HAL_AMDGPU_MAX_GPU_AGENT - 1].handle,
            gpu_agent.handle);
  EXPECT_EQ(topology.gpu_cpu_map[IREE_HAL_AMDGPU_MAX_GPU_AGENT - 1], 0);
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
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_EQ(topology.all_agent_count, 2);
  EXPECT_EQ(topology.cpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_queue_count,
            IREE_HAL_AMDGPU_DEFAULT_GPU_AGENT_QUEUE_COUNT);
  EXPECT_EQ(topology.gpu_cpu_map[0], 0);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

TEST_F(TopologyTest, InitializeFromPathTwoOrdinals) {
  iree_hal_amdgpu_topology_t topology;
  iree_status_t status = iree_hal_amdgpu_topology_initialize_from_path(
      &libhsa, IREE_SV("0,1"), &topology);
  if (!iree_status_is_ok(status)) {
    iree_status_code_t status_code = iree_status_code(status);
    if (status_code == IREE_STATUS_INVALID_ARGUMENT) {
      iree_status_free(status);
      GTEST_SKIP() << "fewer than two visible GPU agents";
      return;
    }
    IREE_ASSERT_OK(status);
  }
  ExpectTopologyHasTwoGpus(topology);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

TEST_F(TopologyTest, InitializeFromPathTwoDefaultGpuUuidsVerifies) {
  iree_hal_amdgpu_topology_t defaults;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &defaults));
  if (defaults.gpu_agent_count < 2) {
    iree_hal_amdgpu_topology_deinitialize(&defaults);
    GTEST_SKIP() << "fewer than two compatible GPU agents";
    return;
  }

  std::string path;
  IREE_ASSERT_OK(
      AppendAgentUuidPathFragment(&libhsa, defaults.gpu_agents[0], &path));
  IREE_ASSERT_OK(
      AppendAgentUuidPathFragment(&libhsa, defaults.gpu_agents[1], &path));

  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_path(
      &libhsa, iree_make_cstring_view(path.c_str()), &topology));
  ExpectTopologyHasTwoGpus(topology);
  IREE_EXPECT_OK(iree_hal_amdgpu_topology_verify(&topology, &libhsa));
  iree_hal_amdgpu_topology_deinitialize(&topology);
  iree_hal_amdgpu_topology_deinitialize(&defaults);
}

// Tests that initialize_from_gpu_agent_mask with a 0 mask is the same as
// initializing from defaults.
TEST_F(TopologyTest, InitializeFromGPUAgentMask0) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(
      &libhsa, 0ull, &topology));
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

// Tests initializing a single device with the given ordinal mask.
TEST_F(TopologyTest, InitializeFromGPUAgentMask1) {
  iree_hal_amdgpu_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(
      &libhsa, 1ull << 0, &topology));
  if (topology.gpu_agent_count == 0) {
    GTEST_SKIP() << "no GPU agents found";
    return;
  }
  EXPECT_EQ(topology.all_agent_count, 2);
  EXPECT_EQ(topology.cpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_count, 1);
  EXPECT_EQ(topology.gpu_agent_queue_count,
            IREE_HAL_AMDGPU_DEFAULT_GPU_AGENT_QUEUE_COUNT);
  EXPECT_EQ(topology.gpu_cpu_map[0], 0);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

TEST_F(TopologyTest, InitializeFromGPUAgentMaskTwoDevices) {
  iree_hal_amdgpu_topology_t topology;
  iree_status_t status =
      iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(&libhsa, 0x3ull,
                                                              &topology);
  if (!iree_status_is_ok(status)) {
    iree_status_code_t status_code = iree_status_code(status);
    if (status_code == IREE_STATUS_OUT_OF_RANGE) {
      iree_status_free(status);
      GTEST_SKIP() << "fewer than two visible GPU agents";
      return;
    }
    IREE_ASSERT_OK(status);
  }
  ExpectTopologyHasTwoGpus(topology);
  iree_hal_amdgpu_topology_deinitialize(&topology);
}

}  // namespace
}  // namespace iree::hal::amdgpu
