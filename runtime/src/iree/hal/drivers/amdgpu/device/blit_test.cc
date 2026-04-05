// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/blit.h"

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

constexpr uint64_t kFillX1KernelObject = 0xF11u;
constexpr uint64_t kFillX2KernelObject = 0xF12u;
constexpr uint64_t kFillX4KernelObject = 0xF14u;
constexpr uint64_t kFillX8KernelObject = 0xF18u;
constexpr uint64_t kFillBlockX16KernelObject = 0xF160u;
constexpr uint64_t kCopyX1KernelObject = 0xC11u;
constexpr uint64_t kCopyBlockX16KernelObject = 0xC160u;

static iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs(
    uint64_t kernel_object, uint16_t setup, uint16_t workgroup_size_x,
    uint32_t private_segment_size, uint32_t group_segment_size) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = kernel_object;
  kernel_args.kernarg_size = 24;
  kernel_args.kernarg_alignment = 8;
  kernel_args.setup = setup;
  kernel_args.workgroup_size[0] = workgroup_size_x;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.private_segment_size = private_segment_size;
  kernel_args.group_segment_size = group_segment_size;
  return kernel_args;
}

static iree_hal_amdgpu_device_kernels_t MakeKernels() {
  iree_hal_amdgpu_device_kernels_t kernels = {};
  kernels.iree_hal_amdgpu_device_buffer_fill_x1 =
      MakeKernelArgs(kFillX1KernelObject, 1, 32, 4, 8);
  kernels.iree_hal_amdgpu_device_buffer_fill_x2 =
      MakeKernelArgs(kFillX2KernelObject, 2, 32, 5, 9);
  kernels.iree_hal_amdgpu_device_buffer_fill_x4 =
      MakeKernelArgs(kFillX4KernelObject, 3, 32, 6, 10);
  kernels.iree_hal_amdgpu_device_buffer_fill_x8 =
      MakeKernelArgs(kFillX8KernelObject, 4, 32, 7, 11);
  kernels.iree_hal_amdgpu_device_buffer_fill_block_x16 =
      MakeKernelArgs(kFillBlockX16KernelObject, 5, 32, 8, 12);
  kernels.iree_hal_amdgpu_device_buffer_copy_x1 =
      MakeKernelArgs(kCopyX1KernelObject, 6, 32, 9, 13);
  kernels.iree_hal_amdgpu_device_buffer_copy_block_x16 =
      MakeKernelArgs(kCopyBlockX16KernelObject, 7, 32, 10, 14);
  return kernels;
}

TEST(BlitTest, FillEmplaceSelectsBlockFillForAlignedTransfer) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs = {};

  ASSERT_TRUE(iree_hal_amdgpu_device_buffer_fill_emplace(
      &context, &packet, (void*)0x2000, /*length=*/512,
      /*pattern=*/0xABu,
      /*pattern_length=*/1, &kernargs));

  EXPECT_EQ(packet.setup, 5);
  EXPECT_EQ(packet.workgroup_size[0], 32);
  EXPECT_EQ(packet.workgroup_size[1], 1);
  EXPECT_EQ(packet.workgroup_size[2], 1);
  EXPECT_EQ(packet.grid_size[0], 4);
  EXPECT_EQ(packet.grid_size[1], 1);
  EXPECT_EQ(packet.grid_size[2], 1);
  EXPECT_EQ(packet.private_segment_size, 8);
  EXPECT_EQ(packet.group_segment_size, 12);
  EXPECT_EQ(packet.kernel_object, kFillBlockX16KernelObject);
  EXPECT_EQ(packet.kernarg_address, &kernargs);
  EXPECT_EQ(packet.completion_signal.handle, iree_hsa_signal_null().handle);

  EXPECT_EQ(kernargs.target_ptr, (void*)0x2000);
  EXPECT_EQ(kernargs.element_length, 32u);
  EXPECT_EQ(kernargs.pattern, 0xABABABABABABABABull);
}

TEST(BlitTest, FillEmplaceMasksPatternToDeclaredWidth) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs = {};

  ASSERT_TRUE(iree_hal_amdgpu_device_buffer_fill_emplace(
      &context, &packet, (void*)0x2000, /*length=*/512,
      /*pattern=*/0x1ABu,
      /*pattern_length=*/1, &kernargs));

  EXPECT_EQ(packet.kernel_object, kFillBlockX16KernelObject);
  EXPECT_EQ(kernargs.pattern, 0xABABABABABABABABull);
}

TEST(BlitTest, FillEmplaceRejectsUnsupportedPatternLengthWithoutMutation) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  packet.setup = 0x55AAu;
  packet.kernel_object = 0xDEADBEEFu;
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs = {
      /*.target_ptr=*/(void*)0x1234,
      /*.element_length=*/7,
      /*.pattern=*/0x99,
  };

  EXPECT_FALSE(iree_hal_amdgpu_device_buffer_fill_emplace(
      &context, &packet, (void*)0x2000, /*length=*/12,
      /*pattern=*/0xABCDu,
      /*pattern_length=*/3, &kernargs));

  EXPECT_EQ(packet.setup, 0x55AAu);
  EXPECT_EQ(packet.kernel_object, 0xDEADBEEFu);
  EXPECT_EQ(kernargs.target_ptr, (void*)0x1234);
  EXPECT_EQ(kernargs.element_length, 7u);
  EXPECT_EQ(kernargs.pattern, 0x99u);
}

TEST(BlitTest, FillEmplaceRejectsMisalignedPatternTransferWithoutMutation) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  packet.setup = 0x55AAu;
  packet.kernel_object = 0xDEADBEEFu;
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs = {
      /*.target_ptr=*/(void*)0x1234,
      /*.element_length=*/7,
      /*.pattern=*/0x99,
  };

  EXPECT_FALSE(iree_hal_amdgpu_device_buffer_fill_emplace(
      &context, &packet, (void*)0x2002, /*length=*/16,
      /*pattern=*/0xABCDu,
      /*pattern_length=*/4, &kernargs));

  EXPECT_EQ(packet.setup, 0x55AAu);
  EXPECT_EQ(packet.kernel_object, 0xDEADBEEFu);
  EXPECT_EQ(kernargs.target_ptr, (void*)0x1234);
  EXPECT_EQ(kernargs.element_length, 7u);
  EXPECT_EQ(kernargs.pattern, 0x99u);
}

TEST(BlitTest, CopyEmplaceSelectsBlockCopyForAlignedTransfer) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs = {};

  ASSERT_TRUE(iree_hal_amdgpu_device_buffer_copy_emplace(
      &context, &packet, (const void*)0x4000, (void*)0x8000,
      /*length=*/256, &kernargs));

  EXPECT_EQ(packet.setup, 7);
  EXPECT_EQ(packet.workgroup_size[0], 32);
  EXPECT_EQ(packet.workgroup_size[1], 1);
  EXPECT_EQ(packet.workgroup_size[2], 1);
  EXPECT_EQ(packet.grid_size[0], 2);
  EXPECT_EQ(packet.grid_size[1], 1);
  EXPECT_EQ(packet.grid_size[2], 1);
  EXPECT_EQ(packet.private_segment_size, 10);
  EXPECT_EQ(packet.group_segment_size, 14);
  EXPECT_EQ(packet.kernel_object, kCopyBlockX16KernelObject);
  EXPECT_EQ(packet.kernarg_address, &kernargs);
  EXPECT_EQ(packet.completion_signal.handle, iree_hsa_signal_null().handle);

  EXPECT_EQ(kernargs.source_ptr, (const void*)0x4000);
  EXPECT_EQ(kernargs.target_ptr, (void*)0x8000);
  EXPECT_EQ(kernargs.element_length, 16u);
}

TEST(BlitTest, CopyEmplaceFallsBackToByteCopyForUnalignedTransfer) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs = {};

  ASSERT_TRUE(iree_hal_amdgpu_device_buffer_copy_emplace(
      &context, &packet, (const void*)0x4001, (void*)0x8002,
      /*length=*/17, &kernargs));

  EXPECT_EQ(packet.setup, 6);
  EXPECT_EQ(packet.grid_size[0], 17);
  EXPECT_EQ(packet.grid_size[1], 1);
  EXPECT_EQ(packet.grid_size[2], 1);
  EXPECT_EQ(packet.kernel_object, kCopyX1KernelObject);
  EXPECT_EQ(kernargs.source_ptr, (const void*)0x4001);
  EXPECT_EQ(kernargs.target_ptr, (void*)0x8002);
  EXPECT_EQ(kernargs.element_length, 17u);
}

TEST(BlitTest, CopyEmplaceRejectsTooLargeTransferWithoutMutation) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t context = {
      /*.kernels=*/&kernels,
  };
  iree_hsa_kernel_dispatch_packet_t packet = {};
  packet.setup = 0x55AAu;
  packet.kernel_object = 0xDEADBEEFu;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs = {
      /*.source_ptr=*/(const void*)0x1234,
      /*.target_ptr=*/(void*)0x5678,
      /*.element_length=*/7,
  };

  EXPECT_FALSE(iree_hal_amdgpu_device_buffer_copy_emplace(
      &context, &packet, (const void*)0x4001, (void*)0x8002,
      /*length=*/UINT64_MAX, &kernargs));

  EXPECT_EQ(packet.setup, 0x55AAu);
  EXPECT_EQ(packet.kernel_object, 0xDEADBEEFu);
  EXPECT_EQ(kernargs.source_ptr, (const void*)0x1234);
  EXPECT_EQ(kernargs.target_ptr, (void*)0x5678);
  EXPECT_EQ(kernargs.element_length, 7u);
}

}  // namespace
}  // namespace iree::hal::amdgpu
