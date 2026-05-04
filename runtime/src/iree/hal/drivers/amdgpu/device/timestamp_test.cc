// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/timestamp.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "iree/hal/drivers/amdgpu/abi/profile.h"
#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_device_kernel_args_t MakeHarvestKernelArgs() {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = 0x12345678ull;
  kernel_args.setup = 2;
  kernel_args.workgroup_size[0] = 32;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.kernarg_alignment = 16;
  return kernel_args;
}

TEST(TimestampTest, AbiRecordLayoutIsFixed) {
  EXPECT_EQ(sizeof(iree_hal_amdgpu_timestamp_range_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_timestamp_record_header_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_timestamp_record_t), 48u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_dispatch_timestamp_record_t), 64u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_command_buffer_timestamp_record_t, ticks),
            32u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_dispatch_timestamp_record_t, ticks), 48u);
}

TEST(TimestampTest, MakesRecordHeader) {
  iree_hal_amdgpu_timestamp_record_header_t header = {};
  header.record_length = sizeof(iree_hal_amdgpu_dispatch_timestamp_record_t);
  header.version = IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_0;
  header.type = IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_DISPATCH;
  header.record_ordinal = 7;

  EXPECT_EQ(header.record_length,
            sizeof(iree_hal_amdgpu_dispatch_timestamp_record_t));
  EXPECT_EQ(header.version, IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_0);
  EXPECT_EQ(header.type, IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_DISPATCH);
  EXPECT_EQ(header.record_ordinal, 7u);
  EXPECT_EQ(header.reserved0, 0u);
}

TEST(TimestampTest, ComputesHarvestKernargLayout) {
  EXPECT_EQ(iree_hal_amdgpu_device_timestamp_dispatch_harvest_source_offset(),
            16u);
  EXPECT_EQ(iree_hal_amdgpu_device_timestamp_dispatch_harvest_kernarg_length(0),
            16u);
  EXPECT_EQ(iree_hal_amdgpu_device_timestamp_dispatch_harvest_kernarg_length(3),
            64u);
}

TEST(TimestampTest, ProfileDispatchHarvestUsesTimestampRangeTarget) {
  EXPECT_EQ(sizeof(iree_hal_amdgpu_profile_dispatch_harvest_source_t),
            sizeof(iree_hal_amdgpu_dispatch_timestamp_harvest_source_t));
  EXPECT_EQ(sizeof(iree_hal_amdgpu_profile_dispatch_harvest_args_t),
            sizeof(iree_hal_amdgpu_dispatch_timestamp_harvest_args_t));

  iree_hal_amdgpu_profile_dispatch_event_t event = {};
  iree_hal_amdgpu_timestamp_range_t* ticks =
      iree_hal_amdgpu_profile_dispatch_event_ticks(&event);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ticks),
            reinterpret_cast<uintptr_t>(&event.start_tick));

  ticks->start_tick = 11;
  ticks->end_tick = 22;
  EXPECT_EQ(event.start_tick, 11u);
  EXPECT_EQ(event.end_tick, 22u);
}

TEST(TimestampTest, EmplacesDispatchHarvestPacketAndKernargs) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeHarvestKernelArgs();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  packet.header = 0xFFFFu;
  alignas(16) std::array<uint8_t, 256> kernargs = {};
  const uint32_t source_count = 65;

  iree_hal_amdgpu_dispatch_timestamp_harvest_source_t* sources =
      iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
          &kernel_args, source_count, &packet, kernargs.data());
  const auto* args = reinterpret_cast<
      const iree_hal_amdgpu_dispatch_timestamp_harvest_args_t*>(
      kernargs.data());

  EXPECT_EQ(args->sources, sources);
  EXPECT_EQ(args->source_count, source_count);
  EXPECT_EQ(args->reserved0, 0u);
  EXPECT_EQ(
      sources,
      reinterpret_cast<iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*>(
          kernargs.data() +
          iree_hal_amdgpu_device_timestamp_dispatch_harvest_source_offset()));

  EXPECT_EQ(packet.header, 0xFFFFu);
  EXPECT_EQ(packet.setup, 2u);
  EXPECT_EQ(packet.workgroup_size[0], 32u);
  EXPECT_EQ(packet.workgroup_size[1], 1u);
  EXPECT_EQ(packet.workgroup_size[2], 1u);
  EXPECT_EQ(packet.grid_size[0], 96u);
  EXPECT_EQ(packet.grid_size[1], 1u);
  EXPECT_EQ(packet.grid_size[2], 1u);
  EXPECT_EQ(packet.kernel_object, 0x12345678ull);
  EXPECT_EQ(packet.kernarg_address, kernargs.data());
  EXPECT_EQ(packet.completion_signal.handle, iree_hsa_signal_null().handle);
}

}  // namespace
}  // namespace iree::hal::amdgpu
