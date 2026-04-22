// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/elf_format.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

static void StoreU16LE(std::vector<uint8_t>* data, size_t offset,
                       uint16_t value) {
  (*data)[offset + 0] = (uint8_t)value;
  (*data)[offset + 1] = (uint8_t)(value >> 8);
}

static void StoreU32LE(std::vector<uint8_t>* data, size_t offset,
                       uint32_t value) {
  (*data)[offset + 0] = (uint8_t)value;
  (*data)[offset + 1] = (uint8_t)(value >> 8);
  (*data)[offset + 2] = (uint8_t)(value >> 16);
  (*data)[offset + 3] = (uint8_t)(value >> 24);
}

static void StoreU64LE(std::vector<uint8_t>* data, size_t offset,
                       uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    (*data)[offset + i] = (uint8_t)(value >> (i * 8));
  }
}

static std::vector<uint8_t> BuildElf64() {
  constexpr size_t kElf64HeaderSize = 64;
  constexpr size_t kProgramHeaderOffset = 64;
  constexpr size_t kProgramHeaderSize = 56;
  constexpr size_t kSectionHeaderOffset = 160;
  constexpr size_t kSectionHeaderSize = 64;
  constexpr size_t kSectionDataOffset = 512;
  constexpr size_t kSectionDataSize = 96;
  std::vector<uint8_t> data(kSectionDataOffset + kSectionDataSize, 0);

  data[0] = 0x7F;
  data[1] = 'E';
  data[2] = 'L';
  data[3] = 'F';
  data[4] = 2;  // ELFCLASS64
  data[5] = 1;  // ELFDATA2LSB
  data[6] = 1;  // EV_CURRENT
  StoreU16LE(&data, 16, 3);
  StoreU16LE(&data, 18, 224);
  StoreU32LE(&data, 20, 1);
  StoreU64LE(&data, 32, kProgramHeaderOffset);
  StoreU64LE(&data, 40, kSectionHeaderOffset);
  StoreU16LE(&data, 52, kElf64HeaderSize);
  StoreU16LE(&data, 54, kProgramHeaderSize);
  StoreU16LE(&data, 56, 1);
  StoreU16LE(&data, 58, kSectionHeaderSize);
  StoreU16LE(&data, 60, 1);

  StoreU32LE(&data, kSectionHeaderOffset + 0, 1);
  StoreU32LE(&data, kSectionHeaderOffset + 4, 1);
  StoreU64LE(&data, kSectionHeaderOffset + 24, kSectionDataOffset);
  StoreU64LE(&data, kSectionHeaderOffset + 32, kSectionDataSize);
  return data;
}

static iree_const_byte_span_t ByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

TEST(ElfFormatTest, DetectsElfMagic) {
  std::vector<uint8_t> data = BuildElf64();
  EXPECT_TRUE(iree_hal_elf_data_starts_with_magic(ByteSpan(data)));
  EXPECT_TRUE(iree_hal_elf_data_starts_with_magic(
      iree_make_const_byte_span(data.data(), 0)));

  data[0] = 0;
  EXPECT_FALSE(iree_hal_elf_data_starts_with_magic(ByteSpan(data)));
}

TEST(ElfFormatTest, CalculatesKnownSize) {
  std::vector<uint8_t> data = BuildElf64();
  iree_host_size_t size = 0;
  IREE_ASSERT_OK(iree_hal_elf_calculate_size(ByteSpan(data), &size));
  EXPECT_EQ(size, data.size());
}

TEST(ElfFormatTest, CalculatesUnknownSize) {
  std::vector<uint8_t> data = BuildElf64();
  iree_host_size_t size = 0;
  IREE_ASSERT_OK(iree_hal_elf_calculate_size(
      iree_make_const_byte_span(data.data(), 0), &size));
  EXPECT_EQ(size, data.size());
}

TEST(ElfFormatTest, RejectsTruncatedKnownSize) {
  std::vector<uint8_t> data = BuildElf64();
  iree_host_size_t size = 0;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_elf_calculate_size(iree_make_const_byte_span(data.data(), 128),
                                  &size));
}

TEST(ElfFormatTest, RejectsSmallSectionHeaderEntry) {
  std::vector<uint8_t> data = BuildElf64();
  StoreU16LE(&data, 58, 1);
  iree_host_size_t size = 0;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_elf_calculate_size(ByteSpan(data), &size));
}

}  // namespace
}  // namespace iree::hal
