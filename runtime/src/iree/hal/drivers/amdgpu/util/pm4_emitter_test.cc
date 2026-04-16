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

}  // namespace
