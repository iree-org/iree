// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace {

constexpr iree_hal_amdgpu_vendor_packet_capability_flags_t
    kBarrierCapabilities =
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM;

TEST(PM4BarrierTest, MapsFenceScopesToGfx10GcrControl) {
  EXPECT_EQ(iree_hal_amdgpu_pm4_barrier_gcr_cntl_for_scopes_gfx10(
                IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
            0u);

  const uint32_t agent_gcr_cntl =
      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLI_INV_ALL |
      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLK_INV |
      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLV_INV |
      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GL1_INV;
  EXPECT_EQ(iree_hal_amdgpu_pm4_barrier_gcr_cntl_for_scopes_gfx10(
                IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_NONE),
            agent_gcr_cntl);

  EXPECT_EQ(iree_hal_amdgpu_pm4_barrier_gcr_cntl_for_scopes_gfx10(
                IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_SYSTEM),
            IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_CNTL_CONSERVATIVE);
}

TEST(PM4BarrierTest, EmitsExecutionOnlyBarrier) {
  uint32_t dwords[IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT];
  std::memset(dwords, 0xCC, sizeof(dwords));
  uint32_t dword_count = 0;

  EXPECT_TRUE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_ARRAYSIZE(dwords), dwords, &dword_count));

  EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT);
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_EVENT_WRITE,
                           IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT));
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_TYPE_CS_PARTIAL_FLUSH |
                IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_INDEX_CS_PARTIAL_FLUSH);
}

TEST(PM4BarrierTest, EmitsScopedExecutionBarrier) {
  uint32_t dwords[IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT];
  std::memset(dwords, 0xCC, sizeof(dwords));
  uint32_t dword_count = 0;

  EXPECT_TRUE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_ARRAYSIZE(dwords), dwords, &dword_count));

  EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT);
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_EVENT_WRITE,
                           IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT));
  EXPECT_EQ(dwords[2], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_ACQUIRE_MEM,
                           IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_DWORD_COUNT));
  EXPECT_EQ(dwords[3], 0u);
  EXPECT_EQ(dwords[4], IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_COHER_SIZE);
  EXPECT_EQ(dwords[5], IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_COHER_SIZE_HI);
  EXPECT_EQ(dwords[6], 0u);
  EXPECT_EQ(dwords[7], 0u);
  EXPECT_EQ(dwords[8], IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_POLL_INTERVAL);
  EXPECT_EQ(dwords[9], IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_CNTL_CONSERVATIVE);
}

TEST(PM4BarrierTest, EmitsConservativeFixupToIbVisibilityBarrier) {
  uint32_t dwords[IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT];
  std::memset(dwords, 0xCC, sizeof(dwords));
  uint32_t dword_count = 0;

  EXPECT_TRUE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_ARRAYSIZE(dwords), dwords, &dword_count));

  EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT);
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_TYPE_CS_PARTIAL_FLUSH |
                IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_INDEX_CS_PARTIAL_FLUSH);
  EXPECT_EQ(dwords[9], IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_CNTL_CONSERVATIVE);
}

TEST(PM4BarrierTest, RejectsInvalidArgumentsWithoutWriting) {
  uint32_t dwords[IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT];
  std::memset(dwords, 0xCC, sizeof(dwords));
  uint32_t dword_count = 1234;

  EXPECT_FALSE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_ARRAYSIZE(dwords), dwords, &dword_count));
  EXPECT_EQ(dword_count, 0u);
  EXPECT_EQ(dwords[0], 0xCCCCCCCCu);

  EXPECT_FALSE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      (iree_hsa_fence_scope_t)3, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_ARRAYSIZE(dwords), dwords, &dword_count));
  EXPECT_EQ(dwords[0], 0xCCCCCCCCu);

  EXPECT_FALSE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE,
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_ARRAYSIZE(dwords), dwords,
      &dword_count));
  EXPECT_EQ(dwords[0], 0xCCCCCCCCu);

  EXPECT_FALSE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT - 1, dwords,
      &dword_count));
  EXPECT_EQ(dwords[0], 0xCCCCCCCCu);
}

}  // namespace
