// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_program.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

struct PM4ProgramTest : public ::testing::Test {
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
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0 || topology.cpu_agent_count == 0) {
      GTEST_SKIP() << "CPU and GPU agents are required, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t PM4ProgramTest::host_allocator;
iree_hal_amdgpu_libhsa_t PM4ProgramTest::libhsa;
iree_hal_amdgpu_topology_t PM4ProgramTest::topology;

TEST_F(PM4ProgramTest, InitializePersistentProgram) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));

  uint32_t source_dwords[32] = {0};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(source_dwords); ++i) {
    source_dwords[i] = 0xC0DEC000u + i;
  }

  iree_hal_amdgpu_pm4_program_t program = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, memory_pool, source_dwords,
      IREE_ARRAYSIZE(source_dwords), &program));

  EXPECT_EQ(program.libhsa, &libhsa);
  EXPECT_EQ(program.memory_pool.handle, memory_pool.handle);
  ASSERT_NE(program.dwords, nullptr);
  EXPECT_EQ(program.dword_count, IREE_ARRAYSIZE(source_dwords));
  EXPECT_EQ(program.byte_length, sizeof(source_dwords));
  EXPECT_EQ(std::memcmp(program.dwords, source_dwords, sizeof(source_dwords)),
            0);

  iree_hsa_amd_aql_pm4_ib_packet_t packet = {};
  uint16_t setup = 0;
  iree_hal_amdgpu_aql_packet_control_t packet_control =
      iree_hal_amdgpu_aql_packet_control_barrier_system();
  uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &packet, program.dwords, program.dword_count, packet_control,
      iree_hsa_signal_null(), &setup);

  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packet.ib_jump_cmd[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER, 4));
  uintptr_t dword_address = (uintptr_t)program.dwords;
  EXPECT_EQ(packet.ib_jump_cmd[1], iree_hal_amdgpu_pm4_addr_lo(dword_address));
  EXPECT_EQ(packet.ib_jump_cmd[2],
            iree_hal_amdgpu_pm4_ib_addr_hi(dword_address));
  EXPECT_EQ(packet.ib_jump_cmd[3], program.dword_count | (1u << 23));
  EXPECT_EQ(packet.dw_cnt_remain, 0xAu);

  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_release(&program));
  EXPECT_EQ(program.dwords, nullptr);
  EXPECT_EQ(program.dword_count, 0u);
  EXPECT_EQ(program.byte_length, 0u);
}

TEST_F(PM4ProgramTest, RejectsInvalidProgramShape) {
  IREE_TRACE_SCOPE();

  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &memory_pool));

  const uint32_t source_dword = 0xC0DEC000u;
  iree_hal_amdgpu_pm4_program_t program = {0};
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_program_initialize(
                  &libhsa, gpu_agent, memory_pool, &source_dword,
                  /*dword_count=*/0, &program)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_program_initialize(
                  &libhsa, gpu_agent, memory_pool, &source_dword,
                  IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT + 1, &program)),
              StatusIs(StatusCode::kOutOfRange));
}

}  // namespace
}  // namespace iree::hal::amdgpu
