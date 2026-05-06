// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

#include <cstring>

#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/testing/gtest.h"

namespace {

TEST(PM4CapabilitiesTest, MemoryWriteDataRequiresPM4IBAndPacketFamily) {
  EXPECT_TRUE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_write_data(
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY));
  EXPECT_FALSE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_write_data(
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB));
  EXPECT_FALSE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_write_data(
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY));
}

TEST(PM4CapabilitiesTest, MemoryCopyDataRequiresPM4IBAndPacketFamily) {
  EXPECT_TRUE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_copy_data(
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY));
  EXPECT_FALSE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_copy_data(
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB));
  EXPECT_FALSE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_copy_data(
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY));
}

TEST(PM4CapabilitiesTest, TimestampRangeRequiresStrategy) {
  EXPECT_FALSE(iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU));
  EXPECT_EQ(iree_hal_amdgpu_pm4_timestamp_range_dword_count(
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE),
            0u);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_timestamp_range_dword_count(
          IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM),
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT);
  EXPECT_EQ(iree_hal_amdgpu_pm4_timestamp_range_dword_count(
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT);
  EXPECT_EQ(iree_hal_amdgpu_pm4_timestamp_range_dword_count(
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT);
}

TEST(PM4CapabilitiesTest, Gfx10PmcProgramsRequireAllPacketFamilies) {
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE;
  EXPECT_TRUE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_gfx10_pmc_programs(
          capabilities));
  capabilities &=
      ~IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK;
  EXPECT_FALSE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_gfx10_pmc_programs(
          capabilities));
}

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
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_remaining(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY - 11u);
}

TEST(PM4EmitterTest, BuilderAppendsRegisterProgrammingPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_event_write_cs_partial_flush(
      &builder));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_set_sh_reg(
      &builder, IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START + 0x34,
      0x11111111u));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_set_uconfig_reg(
      &builder, IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_START + 0x123, 0x22222222u));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 8u);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_EVENT_WRITE,
                           IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT));
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_TYPE_CS_PARTIAL_FLUSH |
                IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_INDEX_CS_PARTIAL_FLUSH);
  EXPECT_EQ(dwords[2], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG,
                           IREE_HAL_AMDGPU_PM4_SET_REGISTER_DWORD_COUNT));
  EXPECT_EQ(dwords[3], 0x34u);
  EXPECT_EQ(dwords[4], 0x11111111u);
  EXPECT_EQ(dwords[5], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_UCONFIG_REG,
                           IREE_HAL_AMDGPU_PM4_SET_REGISTER_DWORD_COUNT));
  EXPECT_EQ(dwords[6], 0x123u);
  EXPECT_EQ(dwords[7], 0x22222222u);
}

TEST(PM4EmitterTest, BuilderAppendsRegisterCopyPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_immediate32_to_register(
      &builder, IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_PERFCOUNTER,
      /*register_address=*/0x2345, /*value=*/0x33333333u,
      IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_NONE));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_register32_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_PERFCOUNTER,
      /*register_address=*/0x3456, target,
      IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_WAIT));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 12u);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_DATA_DWORD_COUNT));
  EXPECT_EQ(dwords[1], IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_IMMEDIATE_DATA |
                           IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_PERFCOUNTER);
  EXPECT_EQ(dwords[2], 0x33333333u);
  EXPECT_EQ(dwords[3], 0u);
  EXPECT_EQ(dwords[4], 0x2345u);
  EXPECT_EQ(dwords[5], 0u);
  EXPECT_EQ(dwords[6], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_DATA_DWORD_COUNT));
  EXPECT_EQ(dwords[7],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_PERFCOUNTER |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(dwords[8], 0x3456u);
  EXPECT_EQ(dwords[9], 0u);
  EXPECT_EQ(dwords[10], 0x9ABCDEF0u);
  EXPECT_EQ(dwords[11], 0x12345678u);
}

TEST(PM4EmitterTest, BuilderRejectsInvalidRegisterPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_set_sh_reg(
      &builder, IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START - 1, 0x11111111u));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_set_uconfig_reg(
      &builder, IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_START - 1, 0x22222222u));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_immediate32_to_register(
      &builder, (iree_hal_amdgpu_pm4_register_space_t)7,
      /*register_address=*/0x2345, /*value=*/0x33333333u,
      IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_NONE));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_immediate32_to_register(
      &builder, IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_PERFCOUNTER,
      /*register_address=*/0x2345, /*value=*/0x33333333u,
      (iree_hal_amdgpu_pm4_write_confirmation_t)7));
  void* unaligned_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEFull));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_register32_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_MEM_MAPPED_REGISTER,
      /*register_address=*/0x3456, unaligned_target,
      IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_NONE));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 0u);
}

TEST(PM4EmitterTest, BuilderAppendsMemoryWritePackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_write_data32(
      &builder, target, /*value=*/0xAABBCCDDu));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_write_data64(
      &builder, target, /*value=*/0x1122334455667788ull));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 11u);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                           /*dword_count=*/5));
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(dwords[2], 0x9ABCDEF0u);
  EXPECT_EQ(dwords[3], 0x12345678u);
  EXPECT_EQ(dwords[4], 0xAABBCCDDu);
  EXPECT_EQ(dwords[5], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                           /*dword_count=*/6));
  EXPECT_EQ(dwords[9], 0x55667788u);
  EXPECT_EQ(dwords[10], 0x11223344u);
}

TEST(PM4EmitterTest, BuilderAppendsMemoryCopyPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  const void* source = reinterpret_cast<const void*>(
      static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_data32(&builder, source,
                                                              target));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_data64(&builder, source,
                                                              target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 12u);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           /*dword_count=*/6));
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(dwords[2], 0x9ABCDEF0u);
  EXPECT_EQ(dwords[3], 0x12345678u);
  EXPECT_EQ(dwords[4], 0x87654320u);
  EXPECT_EQ(dwords[5], 0x0FEDCBA9u);
  EXPECT_EQ(dwords[6], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           /*dword_count=*/6));
  EXPECT_EQ(dwords[7],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
}

TEST(PM4EmitterTest, BuilderAppendsProfiledMemoryWriteInOneIB) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  uint64_t start_tick = 0;
  uint64_t end_tick = 0;
  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_start_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      &start_tick));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_write_data64(
      &builder, target, /*value=*/0x1122334455667788ull));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      &end_tick));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 18u);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
  EXPECT_EQ(dwords[6], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                           /*dword_count=*/6));
  EXPECT_EQ(dwords[12], iree_hal_amdgpu_pm4_make_header(
                            IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                            IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
}

TEST(PM4EmitterTest, EmitsWriteDataMemoryPackets) {
  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  iree_hal_amdgpu_pm4_ib_slot_t slot;

  uint32_t dword_count = iree_hal_amdgpu_pm4_emit_write_data32(
      &slot, target, /*value=*/0xAABBCCDDu);
  EXPECT_EQ(dword_count, 5u);
  EXPECT_EQ(slot.dwords[0], iree_hal_amdgpu_pm4_make_header(
                                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                                /*dword_count=*/5));
  EXPECT_EQ(slot.dwords[1],
            IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(slot.dwords[2], 0x9ABCDEF0u);
  EXPECT_EQ(slot.dwords[3], 0x12345678u);
  EXPECT_EQ(slot.dwords[4], 0xAABBCCDDu);

  dword_count = iree_hal_amdgpu_pm4_emit_write_data64(
      &slot, target, /*value=*/0x1122334455667788ull);
  EXPECT_EQ(dword_count, 6u);
  EXPECT_EQ(slot.dwords[0], iree_hal_amdgpu_pm4_make_header(
                                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA,
                                /*dword_count=*/6));
  EXPECT_EQ(slot.dwords[1],
            IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(slot.dwords[2], 0x9ABCDEF0u);
  EXPECT_EQ(slot.dwords[3], 0x12345678u);
  EXPECT_EQ(slot.dwords[4], 0x55667788u);
  EXPECT_EQ(slot.dwords[5], 0x11223344u);
}

TEST(PM4EmitterTest, EmitsCopyDataMemoryPackets) {
  const void* source = reinterpret_cast<const void*>(
      static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));
  iree_hal_amdgpu_pm4_ib_slot_t slot;

  uint32_t dword_count =
      iree_hal_amdgpu_pm4_emit_copy_data32(&slot, source, target);
  EXPECT_EQ(dword_count, 6u);
  EXPECT_EQ(slot.dwords[0], iree_hal_amdgpu_pm4_make_header(
                                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                                /*dword_count=*/6));
  EXPECT_EQ(slot.dwords[1],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(slot.dwords[2], 0x9ABCDEF0u);
  EXPECT_EQ(slot.dwords[3], 0x12345678u);
  EXPECT_EQ(slot.dwords[4], 0x87654320u);
  EXPECT_EQ(slot.dwords[5], 0x0FEDCBA9u);

  dword_count = iree_hal_amdgpu_pm4_emit_copy_data64(&slot, source, target);
  EXPECT_EQ(dword_count, 6u);
  EXPECT_EQ(slot.dwords[0], iree_hal_amdgpu_pm4_make_header(
                                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                                /*dword_count=*/6));
  EXPECT_EQ(slot.dwords[1],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION);
  EXPECT_EQ(slot.dwords[2], 0x9ABCDEF0u);
  EXPECT_EQ(slot.dwords[3], 0x12345678u);
  EXPECT_EQ(slot.dwords[4], 0x87654320u);
  EXPECT_EQ(slot.dwords[5], 0x0FEDCBA9u);
}

TEST(PM4EmitterTest, EmitsWaitRegMem64Packet) {
  iree_amd_signal_t signal_abi = {};
  iree_hsa_signal_t epoch_signal = {};
  epoch_signal.handle = reinterpret_cast<uint64_t>(&signal_abi);
  iree_hal_amdgpu_pm4_ib_slot_t slot;

  const uint32_t dword_count = iree_hal_amdgpu_pm4_emit_wait_reg_mem64(
      &slot, epoch_signal, /*compare_value=*/0x1122334455667788ll,
      /*mask=*/0x7FFFFFFFFFFFFFFFll);
  EXPECT_EQ(dword_count, 9u);
  EXPECT_EQ(slot.dwords[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64,
                /*dword_count=*/9));
  EXPECT_EQ(slot.dwords[1],
            iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
                IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_LESS_THAN,
                IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY,
                IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM));

  const uintptr_t value_address =
      reinterpret_cast<uintptr_t>(&signal_abi.value);
  EXPECT_EQ(slot.dwords[2], iree_hal_amdgpu_pm4_addr_lo_8(value_address));
  EXPECT_EQ(slot.dwords[3], iree_hal_amdgpu_pm4_addr_hi(value_address));
  EXPECT_EQ(slot.dwords[4], 0x55667788u);
  EXPECT_EQ(slot.dwords[5], 0x11223344u);
  EXPECT_EQ(slot.dwords[6], 0xFFFFFFFFu);
  EXPECT_EQ(slot.dwords[7], 0x7FFFFFFFu);
  EXPECT_EQ(slot.dwords[8],
            4u | IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE);
}

TEST(PM4EmitterTest, TimestampControlsMatchAqlprofileFamilies) {
  const uint32_t common =
      IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_GPU_CLOCK_COUNT |
      IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS;
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_copy_timestamp_control(
          IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM),
      common | IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_CACHE_POLICY_STREAM |
          IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
          IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_CACHE_POLICY_STREAM);
  EXPECT_EQ(iree_hal_amdgpu_pm4_copy_timestamp_control(
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU),
            common | IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2);
  EXPECT_EQ(iree_hal_amdgpu_pm4_copy_timestamp_control(
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU),
            common | IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_TEMPORAL_LU |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_TEMPORAL_LU);
  EXPECT_EQ(iree_hal_amdgpu_pm4_copy_timestamp_control(
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE),
            0u);
}

TEST(PM4EmitterTest, BuilderAppendsTimestampPackets) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* start_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  void* end_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));

  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      start_target));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      end_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT);
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_remaining(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY -
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
  EXPECT_EQ(dwords[1],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_GPU_CLOCK_COUNT |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_CACHE_POLICY_STREAM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_CACHE_POLICY_STREAM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS);
  EXPECT_EQ(dwords[2], 0u);
  EXPECT_EQ(dwords[3], 0u);
  EXPECT_EQ(dwords[4], 0x9ABCDEF0u);
  EXPECT_EQ(dwords[5], 0x12345678u);
  EXPECT_EQ(dwords[6], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
  EXPECT_EQ(dwords[7],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_GPU_CLOCK_COUNT |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_CACHE_POLICY_STREAM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_CACHE_POLICY_STREAM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS);
  EXPECT_EQ(dwords[8], 0u);
  EXPECT_EQ(dwords[9], 0u);
  EXPECT_EQ(dwords[10], 0x87654320u);
  EXPECT_EQ(dwords[11], 0x0FEDCBA9u);
}

TEST(PM4EmitterTest, BuilderAppendsCopyClockTimestampEndStrategy) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT);
  EXPECT_EQ(slot.dwords[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
  EXPECT_EQ(slot.dwords[1],
            IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_GPU_CLOCK_COUNT |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_CACHE_POLICY_STREAM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_CACHE_POLICY_STREAM |
                IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS);
  EXPECT_EQ(slot.dwords[4], 0x87654320u);
  EXPECT_EQ(slot.dwords[5], 0x0FEDCBA9u);
}

TEST(PM4EmitterTest, BuilderAppendsTimestampRangeAtomically) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  void* start_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  void* end_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));

  EXPECT_TRUE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      start_target, end_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT);

  const uint32_t* dwords = slot.dwords;
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
  EXPECT_EQ(dwords[6], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                           IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
}

TEST(PM4EmitterTest, EmitsTimestampAqlPackets) {
  iree_hal_amdgpu_aql_packet_control_t packet_control =
      iree_hal_amdgpu_aql_packet_control_barrier_system();
  const iree_hsa_signal_t completion_signal = {0x12345678ull};
  uint64_t start_tick = 0;
  uint64_t end_tick = 0;

  iree_hsa_amd_aql_pm4_ib_packet_t packet = {};
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  uint16_t setup = 0;
  uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_start(
      &packet, &slot, packet_control,
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      &start_tick, &setup);
  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packet.completion_signal.handle, iree_hsa_signal_null().handle);
  EXPECT_EQ(slot.dwords[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));

  packet = {};
  header = iree_hal_amdgpu_aql_emit_timestamp_end(
      &packet, &slot, packet_control,
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      completion_signal, &end_tick, &setup);
  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packet.completion_signal.handle, completion_signal.handle);
  EXPECT_EQ(slot.dwords[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));

  packet = {};
  header = iree_hal_amdgpu_aql_emit_timestamp_range(
      &packet, &slot, packet_control,
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      completion_signal, &start_tick, &end_tick, &setup);
  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packet.completion_signal.handle, completion_signal.handle);
  EXPECT_EQ(slot.dwords[0],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
  EXPECT_EQ(slot.dwords[6],
            iree_hal_amdgpu_pm4_make_header(
                IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
                IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT));
}

TEST(PM4EmitterTest, BuilderRejectsTimestampRangeWithoutPartialAppend) {
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);

  uint32_t* prefix = iree_hal_amdgpu_pm4_ib_builder_append_dwords(
      &builder, IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY -
                    IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT + 1);
  ASSERT_NE(prefix, nullptr);
  void* start_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  void* end_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x0FEDCBA987654320ull));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      start_target, end_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY -
                IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT + 1);

  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);
  void* unaligned_start =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEFull));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      unaligned_start, end_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 0u);
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
  void* aligned_target =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x123456789ABCDEF0ull));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      unaligned_target));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      unaligned_target));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE, aligned_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), 0u);

  uint32_t* prefix = iree_hal_amdgpu_pm4_ib_builder_append_dwords(
      &builder, IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY -
                    IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT + 1);
  ASSERT_NE(prefix, nullptr);
  EXPECT_FALSE(iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
      &builder, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
      aligned_target));
  EXPECT_EQ(iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
            IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY -
                IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT + 1);
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
