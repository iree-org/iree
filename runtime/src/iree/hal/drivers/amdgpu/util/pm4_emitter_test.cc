// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace {

TEST(PM4EmitterTest, BuilderInitializesSlot) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  std::memset(&slot, 0xCC, sizeof(slot));

  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  EXPECT_EQ(builder.slot, &slot);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 0u);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_remaining(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY);
  for (uint32_t i = 0; i < IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY; ++i) {
    EXPECT_EQ(slot.dwords[i], 0u);
  }
}

TEST(PM4EmitterTest, BuilderAppendsPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  uint32_t* write_data_packet = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      &builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
      /*dword_count=*/5);
  ASSERT_NE(write_data_packet, nullptr);
  EXPECT_EQ(write_data_packet, &slot.dwords[0]);
  EXPECT_EQ(write_data_packet[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                /*dword_count=*/5));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 5u);

  uint32_t* copy_data_packet = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      &builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
      /*dword_count=*/6);
  ASSERT_NE(copy_data_packet, nullptr);
  EXPECT_EQ(copy_data_packet, &slot.dwords[5]);
  EXPECT_EQ(copy_data_packet[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                /*dword_count=*/6));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 11u);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_remaining(&builder), 5u);
}

TEST(PM4EmitterTest, BuilderAppendsTimestampPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* copy_timestamp_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  void* release_timestamp_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));

  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
      &builder, copy_timestamp_target));
  EXPECT_TRUE(
      iree_hal_amdgpu_pm4_ib_builder_emit_release_mem_timestamp_to_memory(
          &builder, release_timestamp_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 14u);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_remaining(&builder), 2u);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           /*dword_count=*/6));
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TIMESTAMP |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(dwords[2], 0u);
  EXPECT_EQ(dwords[3], 0u);
  EXPECT_EQ(dwords[4], 0x9ABCDEF0u);
  EXPECT_EQ(dwords[5], 0x12345678u);
  EXPECT_EQ(dwords[6], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_RELEASE_MEM,
                           /*dword_count=*/8));
  EXPECT_EQ(dwords[7],
            IREE_HAL_AMDGPU_PM4_RELEASE_MEM_EVENT_TYPE_BOTTOM_OF_PIPE_TS |
                IREE_HAL_AMDGPU_PM4_RELEASE_MEM_EVENT_INDEX_END_OF_PIPE);
  EXPECT_EQ(dwords[8],
            IREE_HAL_AMDGPU_PM4_RELEASE_MEM_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM |
                IREE_HAL_AMDGPU_PM4_RELEASE_MEM_DATA_SEL_TIMESTAMP);
  EXPECT_EQ(dwords[9], 0x87654320u);
  EXPECT_EQ(dwords[10], 0x0FEDCBA9u);
  EXPECT_EQ(dwords[11], 0u);
  EXPECT_EQ(dwords[12], 0u);
  EXPECT_EQ(dwords[13], 0u);
}

TEST(PM4EmitterTest, BuilderRejectsMalformedAndOverflowPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_append_packet(
                &builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                /*dword_count=*/1),
            nullptr);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 0u);

  uint32_t* full_slot = iree_hal_amdgpu_pm4_ib_builder_append_dwords(
      &builder, IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY);
  ASSERT_NE(full_slot, nullptr);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_append_dwords(&builder, 1), nullptr);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY);
}

TEST(PM4EmitterTest, BuilderRejectsTimestampAlignmentAndOverflow) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* unaligned_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEFull));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
      &builder, unaligned_target));
  EXPECT_FALSE(
      iree_hal_amdgpu_pm4_ib_builder_emit_release_mem_timestamp_to_memory(
          &builder, unaligned_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 0u);

  uint32_t* prefix = iree_hal_amdgpu_pm4_ib_builder_append_dwords(&builder, 10);
  ASSERT_NE(prefix, nullptr);
  void* aligned_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  EXPECT_FALSE(
      iree_hal_amdgpu_pm4_ib_builder_emit_release_mem_timestamp_to_memory(
          &builder, aligned_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 10u);
}

TEST(PM4EmitterTest, EmitsArbitraryPM4IBDwordEnvelope) {
  uint32_t dwords[32] = {0};
  iree_hsa_amd_aql_pm4_ib_packet_t packet = {};

  iree_hal_amdgpu_aql_packet_control_t packet_control =
      iree_hal_amdgpu_aql_packet_control_barrier_system();
  uint16_t setup = 0;
  uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &packet, dwords, IREE_ARRAYSIZE(dwords), packet_control,
      iree_hsa_signal_null(), &setup);

  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packet.ib_jump_cmd[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER,
                /*dword_count=*/4));
  uintptr_t dword_address = reinterpret_cast<uintptr_t>(dwords);
  EXPECT_EQ(packet.ib_jump_cmd[1], iree_hal_amdgpu_pm4_addr_lo(dword_address));
  EXPECT_EQ(packet.ib_jump_cmd[2],
            iree_hal_amdgpu_pm4_ib_addr_hi(dword_address));
  EXPECT_EQ(packet.ib_jump_cmd[3], IREE_ARRAYSIZE(dwords) | (1u << 23));
  EXPECT_EQ(packet.dw_cnt_remain, 0xAu);
}

}  // namespace
