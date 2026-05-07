// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

static iree_hal_amdgpu_kernel_descriptor_t MakeDescriptor(
    uint16_t kernel_code_properties =
        IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
        IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
    uint32_t user_data_dword_count = 2) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {};
  descriptor.group_segment_fixed_size = 1024;
  descriptor.kernarg_size = 64;
  descriptor.kernel_code_entry_byte_offset = 0x140;
  descriptor.compute_pgm_rsrc3 = 0x03020100u;
  descriptor.compute_pgm_rsrc1 = 0x11112222u;
  descriptor.compute_pgm_rsrc2 =
      (user_data_dword_count
       << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) |
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X |
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_MASK |
      (4u << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE_SHIFT);
  descriptor.kernel_code_properties = kernel_code_properties;
  return descriptor;
}

TEST(PM4DispatchTest, KernelDescriptorLayoutMatchesAmdhsaAbi) {
  EXPECT_EQ(sizeof(iree_hal_amdgpu_kernel_descriptor_t), 64u);
  EXPECT_EQ(
      offsetof(iree_hal_amdgpu_kernel_descriptor_t, group_segment_fixed_size),
      0u);
  EXPECT_EQ(
      offsetof(iree_hal_amdgpu_kernel_descriptor_t, private_segment_fixed_size),
      4u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_kernel_descriptor_t, kernarg_size), 8u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_kernel_descriptor_t,
                     kernel_code_entry_byte_offset),
            16u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_kernel_descriptor_t, compute_pgm_rsrc3),
            44u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_kernel_descriptor_t, compute_pgm_rsrc1),
            48u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_kernel_descriptor_t, compute_pgm_rsrc2),
            52u);
  EXPECT_EQ(
      offsetof(iree_hal_amdgpu_kernel_descriptor_t, kernel_code_properties),
      56u);
  EXPECT_EQ(offsetof(iree_hal_amdgpu_kernel_descriptor_t, kernarg_preload),
            58u);
}

TEST(PM4DispatchTest, InitializesLaunchStateFromDescriptor) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor();
  const uint16_t workgroup_size[3] = {256, 2, 1};

  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
      &descriptor, /*kernel_object=*/0x0000123456780000ull, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state));

  const uint32_t expected_program[6] = {
      0x34567801u, 0x12u, 0u, 0u, 0u, 0u,
  };
  EXPECT_EQ(
      std::memcmp(state.program, expected_program, sizeof(expected_program)),
      0);
  EXPECT_EQ(state.resources[0], descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(state.resources[1], descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(state.resource3, descriptor.compute_pgm_rsrc3);
  EXPECT_EQ(state.temporary_ring_size, 0u);
  EXPECT_EQ(state.restart[0], 0u);
  EXPECT_EQ(state.restart[1], 0u);
  EXPECT_EQ(state.restart[2], 0u);
  EXPECT_EQ(state.resource_limits, 0u);

  const uint32_t expected_start_and_threads[8] = {
      0u, 0u, 0u, 256u, 2u, 1u, 0u, 0u,
  };
  EXPECT_EQ(std::memcmp(state.start_and_threads, expected_start_and_threads,
                        sizeof(expected_start_and_threads)),
            0);
  EXPECT_EQ(state.user_data_dword_count, 2u);
  EXPECT_EQ(state.dispatch_initiator,
            IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_COMPUTE_SHADER_EN |
                IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_FORCE_START_AT_000 |
                IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_ORDER_MODE |
                IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_CS_W32_EN);
}

TEST(PM4DispatchTest, InitializesLaunchStateWithPaddedUserDataDwords) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor(
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
          IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
      /*user_data_dword_count=*/15);
  const uint16_t workgroup_size[3] = {64, 1, 1};

  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
      &descriptor, /*kernel_object=*/0x0000123456780000ull, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state));

  EXPECT_EQ(state.user_data_dword_count, 15u);
}

TEST(PM4DispatchTest, EmitsStaticSetupDwords) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor();
  const uint16_t workgroup_size[3] = {128, 1, 1};
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
      &descriptor, /*kernel_object=*/0x0000123456780000ull, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state));

  uint32_t dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT] = {};
  uint32_t dword_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_setup(
      &state, IREE_ARRAYSIZE(dwords), dwords, &dword_count));
  EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT);

  const uint32_t expected[] = {
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 8),
      IREE_HAL_AMDGPU_PM4_COMPUTE_PGM_LO_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.program[0],
      state.program[1],
      state.program[2],
      state.program[3],
      state.program[4],
      state.program[5],
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 4),
      IREE_HAL_AMDGPU_PM4_COMPUTE_PGM_RSRC1_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.resources[0],
      state.resources[1],
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 3),
      IREE_HAL_AMDGPU_PM4_COMPUTE_PGM_RSRC3_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.resource3,
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 3),
      IREE_HAL_AMDGPU_PM4_COMPUTE_TMPRING_SIZE_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.temporary_ring_size,
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 5),
      IREE_HAL_AMDGPU_PM4_COMPUTE_RESTART_X_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.restart[0],
      state.restart[1],
      state.restart[2],
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 3),
      IREE_HAL_AMDGPU_PM4_COMPUTE_RESOURCE_LIMITS_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.resource_limits,
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 10),
      IREE_HAL_AMDGPU_PM4_COMPUTE_START_X_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      state.start_and_threads[0],
      state.start_and_threads[1],
      state.start_and_threads[2],
      state.start_and_threads[3],
      state.start_and_threads[4],
      state.start_and_threads[5],
      state.start_and_threads[6],
      state.start_and_threads[7],
  };
  static_assert(IREE_ARRAYSIZE(expected) ==
                IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT);
  EXPECT_EQ(std::memcmp(dwords, expected, sizeof(expected)), 0);
}

TEST(PM4DispatchTest, EmitsKernargUserDataDwords) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor();
  const uint16_t workgroup_size[3] = {64, 1, 1};
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
      &descriptor, /*kernel_object=*/0x0000123456780000ull, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state));

  uint32_t dwords[4] = {};
  uint32_t dword_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      &state, /*kernarg_address=*/0x00007FFF12345678ull, IREE_ARRAYSIZE(dwords),
      dwords, &dword_count));
  EXPECT_EQ(dword_count, 4u);

  const uint32_t expected[] = {
      iree_hal_amdgpu_pm4_make_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 4),
      IREE_HAL_AMDGPU_PM4_COMPUTE_USER_DATA_0_REGISTER -
          IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START,
      0x12345678u,
      0x00007FFFu,
  };
  EXPECT_EQ(std::memcmp(dwords, expected, sizeof(expected)), 0);
}

TEST(PM4DispatchTest, EmitsPaddedKernargUserDataDwords) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor(
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
          IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
      /*user_data_dword_count=*/15);
  const uint16_t workgroup_size[3] = {64, 1, 1};
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
      &descriptor, /*kernel_object=*/0x0000123456780000ull, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state));

  uint32_t dwords[17] = {};
  uint32_t dword_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      &state, /*kernarg_address=*/0x00007FFF12345678ull, IREE_ARRAYSIZE(dwords),
      dwords, &dword_count));
  EXPECT_EQ(dword_count, 17u);

  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG, 17));
  EXPECT_EQ(dwords[1], IREE_HAL_AMDGPU_PM4_COMPUTE_USER_DATA_0_REGISTER -
                           IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START);
  EXPECT_EQ(dwords[2], 0x12345678u);
  EXPECT_EQ(dwords[3], 0x00007FFFu);
  for (uint32_t i = 4; i < IREE_ARRAYSIZE(dwords); ++i) {
    EXPECT_EQ(dwords[i], 0u);
  }
}

TEST(PM4DispatchTest, RejectsUnsupportedDescriptorShapes) {
  const uint16_t workgroup_size[3] = {64, 1, 1};
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};

  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor();
  descriptor.private_segment_fixed_size = 4;
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kUnimplemented));

  descriptor = MakeDescriptor();
  descriptor.kernarg_preload = 1;
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kUnimplemented));

  descriptor = MakeDescriptor(
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR);
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kUnimplemented));

  descriptor = MakeDescriptor();
  descriptor.compute_pgm_rsrc2 =
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT |
      (2u << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT);
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kUnimplemented));

  descriptor = MakeDescriptor();
  descriptor.compute_pgm_rsrc2 =
      1u << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT;
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kFailedPrecondition));

  descriptor = MakeDescriptor(
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
          IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
      IREE_HAL_AMDGPU_PM4_DISPATCH_USER_DATA_DWORD_CAPACITY + 1);
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(PM4DispatchTest, RejectsInvalidLaunchArguments) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor();
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  const uint16_t valid_workgroup_size[3] = {64, 1, 1};
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0, valid_workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kInvalidArgument));

  const uint16_t invalid_workgroup_size[3] = {64, 0, 1};
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, invalid_workgroup_size,
                  IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state)),
              StatusIs(StatusCode::kInvalidArgument));

  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
                  &descriptor, /*kernel_object=*/0x1000, valid_workgroup_size,
                  /*flags=*/0x80000000u, &state)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(PM4DispatchTest, RejectsInsufficientPacketCapacityWithoutWrites) {
  iree_hal_amdgpu_kernel_descriptor_t descriptor = MakeDescriptor();
  const uint16_t workgroup_size[3] = {64, 1, 1};
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
      &descriptor, /*kernel_object=*/0x0000123456780000ull, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE, &state));

  uint32_t dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT] = {};
  std::memset(dwords, 0xCC, sizeof(dwords));
  uint32_t dword_count = 123;
  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_emit_setup(
                  &state, IREE_ARRAYSIZE(dwords) - 1, dwords, &dword_count)),
              StatusIs(StatusCode::kResourceExhausted));
  EXPECT_EQ(dword_count, 0u);
  for (uint32_t dword : dwords) {
    EXPECT_EQ(dword, 0xCCCCCCCCu);
  }

  EXPECT_THAT(Status(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
                  &state, /*kernarg_address=*/0x1000, /*capacity=*/3, dwords,
                  &dword_count)),
              StatusIs(StatusCode::kResourceExhausted));
  EXPECT_EQ(dword_count, 0u);
  for (uint32_t dword : dwords) {
    EXPECT_EQ(dword, 0xCCCCCCCCu);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
