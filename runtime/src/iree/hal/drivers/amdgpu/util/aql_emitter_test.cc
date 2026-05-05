// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace {

static iree_hsa_signal_t MakeSignal(uint64_t handle) {
  iree_hsa_signal_t signal = {};
  signal.handle = handle;
  return signal;
}

TEST(AQLEmitterTest, EmitsBarrierValuePacketBody) {
  iree_hsa_amd_barrier_value_packet_t packet;
  std::memset(&packet, 0xCC, sizeof(packet));

  const iree_hal_amdgpu_aql_packet_control_t packet_control =
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_SYSTEM);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_value(
      &packet, MakeSignal(0x123456789ABCDEF0ull), IREE_HSA_SIGNAL_CONDITION_LT,
      /*compare_value=*/0x1122334455667788ll,
      /*mask=*/0x7FFFFFFFFFFFFFFFll, packet_control,
      MakeSignal(0x0FEDCBA987654320ull), &setup);

  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_BARRIER_VALUE);

  uint32_t first_dword = 0;
  std::memcpy(&first_dword, &packet, sizeof(first_dword));
  EXPECT_EQ(first_dword, 0xCCCCCCCCu);
  EXPECT_EQ(packet.reserved0, 0u);
  EXPECT_EQ(packet.signal.handle, 0x123456789ABCDEF0ull);
  EXPECT_EQ(packet.value, 0x1122334455667788ll);
  EXPECT_EQ(packet.mask, 0x7FFFFFFFFFFFFFFFll);
  EXPECT_EQ(packet.cond, static_cast<iree_hsa_signal_condition32_t>(
                             IREE_HSA_SIGNAL_CONDITION_LT));
  EXPECT_EQ(packet.reserved1, 0u);
  EXPECT_EQ(packet.reserved2, 0u);
  EXPECT_EQ(packet.reserved3, 0u);
  EXPECT_EQ(packet.completion_signal.handle, 0x0FEDCBA987654320ull);
}

}  // namespace
