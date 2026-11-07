// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/descriptor_range.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_device_size_t kNoRangeLimit = 1ull << 32;

std::string StatusMessage(iree_status_t status) {
  char* buffer = nullptr;
  iree_host_size_t length = 0;
  std::string message;
  iree_allocator_t allocator = iree_allocator_system();
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    message.assign(buffer, length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return message;
}

TEST(DescriptorRangeTest, WholeAllocationUsesAllocatorPadding) {
  // A 6-byte buffer the allocator rounded to 8 bytes.
  iree_device_size_t length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/6, /*owned_length=*/8, /*offset=*/0, /*length=*/6,
      /*minimum_access_bytes=*/4, kNoRangeLimit, &length));
  EXPECT_EQ(length, 8u);
}

TEST(DescriptorRangeTest, ByteGranularBindingKeepsLogicalLength) {
  iree_device_size_t length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/6, /*owned_length=*/8, /*offset=*/0, /*length=*/6,
      /*minimum_access_bytes=*/1, kNoRangeLimit, &length));
  EXPECT_EQ(length, 6u);
}

TEST(DescriptorRangeTest, NarrowedBindingRejectsNeighbourTail) {
  // A 6-byte binding at offset 64 of a 128-byte slab does not own bytes 70-71.
  iree_status_t status = iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/128, /*owned_length=*/128, /*offset=*/64, /*length=*/6,
      /*minimum_access_bytes=*/4, kNoRangeLimit, nullptr);
  ASSERT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  std::string message = StatusMessage(status);
  EXPECT_NE(message.find("8 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("6 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("declare the padding"), std::string::npos) << message;
}

TEST(DescriptorRangeTest, NarrowedBindingExplicitPadding) {
  // The same binding declaring its 2 padding bytes owns them.
  iree_device_size_t length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/128, /*owned_length=*/128, /*offset=*/64, /*length=*/8,
      /*minimum_access_bytes=*/4, kNoRangeLimit, &length));
  EXPECT_EQ(length, 8u);
}

TEST(DescriptorRangeTest, BindingAtViewEndOwnsAllocationTail) {
  // A binding reaching the end of a whole allocation owns the rounded tail.
  iree_device_size_t length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/70, /*owned_length=*/72, /*offset=*/64, /*length=*/6,
      /*minimum_access_bytes=*/4, kNoRangeLimit, &length));
  EXPECT_EQ(length, 8u);
}

TEST(DescriptorRangeTest, SubspanViewDoesNotBorrowParentBytes) {
  // A 6-byte view of a larger buffer owns only its own 6 bytes; padding in the
  // range cannot help, the view itself must cover the unit.
  iree_status_t status = iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/6, /*owned_length=*/6, /*offset=*/0, /*length=*/6,
      /*minimum_access_bytes=*/4, kNoRangeLimit, nullptr);
  ASSERT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  std::string message = StatusMessage(status);
  EXPECT_NE(message.find("whole allocation"), std::string::npos) << message;
}

TEST(DescriptorRangeTest, InsufficientExternalBackingNamesBothSizes) {
  iree_device_size_t length = 0;
  iree_status_t status = iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/10, /*owned_length=*/10, /*offset=*/4, /*length=*/6,
      /*minimum_access_bytes=*/4, kNoRangeLimit, &length);
  ASSERT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  std::string message = StatusMessage(status);
  EXPECT_NE(message.find("8 bytes"), std::string::npos) << message;
  EXPECT_NE(message.find("6 bytes"), std::string::npos) << message;
}

TEST(DescriptorRangeTest, WholeBufferLengthResolvesAgainstView) {
  iree_device_size_t length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/14, /*owned_length=*/16, /*offset=*/4,
      IREE_HAL_WHOLE_BUFFER, /*minimum_access_bytes=*/4, kNoRangeLimit,
      &length));
  EXPECT_EQ(length, 12u);
}

TEST(DescriptorRangeTest, EmptyRangeResolvesToZero) {
  iree_device_size_t length = 1;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/0, /*owned_length=*/4, /*offset=*/0, /*length=*/0,
      /*minimum_access_bytes=*/4, kNoRangeLimit, &length));
  EXPECT_EQ(length, 0u);
}

TEST(DescriptorRangeTest, RejectsRangePastView) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_descriptor_range(
          /*view_length=*/6, /*owned_length=*/8,
          /*offset=*/4, /*length=*/4,
          /*minimum_access_bytes=*/1, kNoRangeLimit, nullptr));
}

TEST(DescriptorRangeTest, RejectsOverflow) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_descriptor_range(
          /*view_length=*/16, /*owned_length=*/16,
          /*offset=*/8, /*length=*/~0ull - 4,
          /*minimum_access_bytes=*/4, kNoRangeLimit, nullptr));
}

TEST(DescriptorRangeTest, RejectsUnalignedOffset) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_resolve_descriptor_range(
          /*view_length=*/16, /*owned_length=*/16,
          /*offset=*/2, /*length=*/6,
          /*minimum_access_bytes=*/4, kNoRangeLimit, nullptr));
}

TEST(DescriptorRangeTest, RejectsRangeOverDeviceLimit) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_descriptor_range(
          /*view_length=*/64, /*owned_length=*/64,
          /*offset=*/0, /*length=*/30,
          /*minimum_access_bytes=*/4, /*max_range=*/30, nullptr));
}

TEST(DescriptorRangeTest, NativeHalfRangeFitsBorrowedView) {
  iree_device_size_t length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/6, /*owned_length=*/6, /*offset=*/0, /*length=*/6,
      /*minimum_access_bytes=*/2, kNoRangeLimit, &length));
  EXPECT_EQ(length, 6u);
}

TEST(DescriptorRangeTest, EmptyUnalignedTailDoesNotNeedPadding) {
  iree_device_size_t length = 1;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_descriptor_range(
      /*view_length=*/6, /*owned_length=*/6, /*offset=*/6, /*length=*/0,
      /*minimum_access_bytes=*/4, kNoRangeLimit, &length));
  EXPECT_EQ(length, 0u);
}

TEST(DescriptorRangeTest, RejectsOverflowWhileRoundingPhysicalEnd) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_descriptor_range(
          /*view_length=*/UINT64_MAX - 1, /*owned_length=*/UINT64_MAX,
          /*offset=*/0, /*length=*/UINT64_MAX - 1,
          /*minimum_access_bytes=*/4, UINT64_MAX, nullptr));
}

TEST(DescriptorRangeTest, BdaWholeAllocationOwnsWordTail) {
  uint64_t address = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_bda_range(
      /*view_length=*/6, /*owned_length=*/8, /*offset=*/0, /*length=*/6,
      /*base_address=*/0x1000, /*minimum_access_bytes=*/4,
      /*minimum_alignment=*/1, /*minimum_length=*/0, &address));
  EXPECT_EQ(address, 0x1000u);
}

TEST(DescriptorRangeTest, BdaBorrowedViewCannotBorrowWordTail) {
  uint64_t address = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_bda_range(6, 6, 0, 6, 0x1000, 4, 1, 0, &address));
  EXPECT_EQ(address, 0u);
}

TEST(DescriptorRangeTest, BdaNativeHalfFitsBorrowedView) {
  uint64_t address = 0;
  IREE_ASSERT_OK(
      iree_hal_vulkan_resolve_bda_range(6, 6, 0, 6, 0x1002, 2, 1, 0, &address));
  EXPECT_EQ(address, 0x1002u);
}

TEST(DescriptorRangeTest, BdaRejectsUnalignedViewAddress) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_resolve_bda_range(8, 8, 0, 8, 0x1002, 4, 1, 0, &address));
}

TEST(DescriptorRangeTest, BdaAlignsFinalAddressWithinUnalignedView) {
  uint64_t address = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_resolve_bda_range(10, 10, 2, 8, 0x1002, 4, 1,
                                                   0, &address));
  EXPECT_EQ(address, 0x1004u);
}

TEST(DescriptorRangeTest, BdaRejectsStrongerShaderAlignment) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_vulkan_resolve_bda_range(8, 8, 0, 8, 0x1004, 4,
                                                          16, 0, &address));
}

TEST(DescriptorRangeTest, BdaRejectsMissingDeviceAddress) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_vulkan_resolve_bda_range(8, 8, 0, 8, 0, 4, 1, 0, &address));
}

TEST(DescriptorRangeTest, BdaEmptyRangeNeedsNoAddressOrAlignment) {
  uint64_t address = 1;
  IREE_ASSERT_OK(
      iree_hal_vulkan_resolve_bda_range(6, 6, 6, 0, 0, 4, 16, 0, &address));
  EXPECT_EQ(address, 0u);
}

TEST(DescriptorRangeTest, BdaChecksPhysicalAddressEnd) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_vulkan_resolve_bda_range(
                            6, 8, 0, 6, UINT64_MAX - 7, 4, 1, 0, &address));
}

TEST(DescriptorRangeTest, BdaChecksAddressAddition) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_vulkan_resolve_bda_range(
                            16, 16, 8, 8, UINT64_MAX - 3, 4, 1, 0, &address));
}

TEST(DescriptorRangeTest, BdaPaddingDoesNotSatisfyLogicalMinimumLength) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_bda_range(6, 8, 0, 6, 0x1000, 4, 1, 8, &address));
}

TEST(DescriptorRangeTest, BdaEmptyRangeDoesNotSatisfyMinimumLength) {
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_resolve_bda_range(0, 4, 0, 0, 0, 4, 1, 4, &address));
}

}  // namespace
