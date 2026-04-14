// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/amdgpu_metadata.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static void AppendByte(std::vector<uint8_t>* output, uint8_t value) {
  output->push_back(value);
}

static void AppendU16BE(std::vector<uint8_t>* output, uint16_t value) {
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)value);
}

static void AppendU32BE(std::vector<uint8_t>* output, uint32_t value) {
  output->push_back((uint8_t)(value >> 24));
  output->push_back((uint8_t)(value >> 16));
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)value);
}

static void AppendU32LE(std::vector<uint8_t>* output, uint32_t value) {
  output->push_back((uint8_t)value);
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)(value >> 16));
  output->push_back((uint8_t)(value >> 24));
}

static void StoreU16LE(std::vector<uint8_t>* output, size_t offset,
                       uint16_t value) {
  (*output)[offset + 0] = (uint8_t)value;
  (*output)[offset + 1] = (uint8_t)(value >> 8);
}

static void StoreU32LE(std::vector<uint8_t>* output, size_t offset,
                       uint32_t value) {
  (*output)[offset + 0] = (uint8_t)value;
  (*output)[offset + 1] = (uint8_t)(value >> 8);
  (*output)[offset + 2] = (uint8_t)(value >> 16);
  (*output)[offset + 3] = (uint8_t)(value >> 24);
}

static void StoreU64LE(std::vector<uint8_t>* output, size_t offset,
                       uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    (*output)[offset + i] = (uint8_t)(value >> (i * 8));
  }
}

static void AppendAligned4Padding(std::vector<uint8_t>* output) {
  while ((output->size() & 3) != 0) output->push_back(0);
}

static void AppendMsgPackMap(std::vector<uint8_t>* output, uint32_t count) {
  if (count < 16) {
    AppendByte(output, (uint8_t)(0x80 | count));
  } else {
    AppendByte(output, 0xDE);
    AppendU16BE(output, (uint16_t)count);
  }
}

static void AppendMsgPackArray(std::vector<uint8_t>* output, uint32_t count) {
  if (count < 16) {
    AppendByte(output, (uint8_t)(0x90 | count));
  } else {
    AppendByte(output, 0xDC);
    AppendU16BE(output, (uint16_t)count);
  }
}

static void AppendMsgPackString(std::vector<uint8_t>* output,
                                iree_string_view_t value) {
  if (value.size < 32) {
    AppendByte(output, (uint8_t)(0xA0 | value.size));
  } else if (value.size <= UINT8_MAX) {
    AppendByte(output, 0xD9);
    AppendByte(output, (uint8_t)value.size);
  } else {
    AppendByte(output, 0xDA);
    AppendU16BE(output, (uint16_t)value.size);
  }
  output->insert(output->end(), value.data, value.data + value.size);
}

static void AppendMsgPackUint(std::vector<uint8_t>* output, uint32_t value) {
  if (value <= 0x7F) {
    AppendByte(output, (uint8_t)value);
  } else if (value <= UINT8_MAX) {
    AppendByte(output, 0xCC);
    AppendByte(output, (uint8_t)value);
  } else if (value <= UINT16_MAX) {
    AppendByte(output, 0xCD);
    AppendU16BE(output, (uint16_t)value);
  } else {
    AppendByte(output, 0xCE);
    AppendU32BE(output, value);
  }
}

static void AppendStringField(std::vector<uint8_t>* output,
                              iree_string_view_t key,
                              iree_string_view_t value) {
  AppendMsgPackString(output, key);
  AppendMsgPackString(output, value);
}

static void AppendUintField(std::vector<uint8_t>* output,
                            iree_string_view_t key, uint32_t value) {
  AppendMsgPackString(output, key);
  AppendMsgPackUint(output, value);
}

static std::vector<uint8_t> BuildKernelMetadata(
    bool out_of_range_arg = false, bool unknown_value_kind = false) {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 3);

  AppendMsgPackString(&output, IREE_SV("amdhsa.version"));
  AppendMsgPackArray(&output, 2);
  AppendMsgPackUint(&output, 1);
  AppendMsgPackUint(&output, 2);

  AppendStringField(&output, IREE_SV("amdhsa.target"),
                    IREE_SV("amdgcn-amd-amdhsa--gfx1100"));

  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 8);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("vector_add"));
  AppendStringField(&output, IREE_SV(".symbol"), IREE_SV("vector_add.kd"));
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 24);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 1024);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 64);
  AppendMsgPackString(&output, IREE_SV(".reqd_workgroup_size"));
  AppendMsgPackArray(&output, 3);
  AppendMsgPackUint(&output, 16);
  AppendMsgPackUint(&output, 4);
  AppendMsgPackUint(&output, 1);

  AppendMsgPackString(&output, IREE_SV(".args"));
  AppendMsgPackArray(&output, 4);

  AppendMsgPackMap(&output, 7);
  AppendUintField(&output, IREE_SV(".offset"), 0);
  AppendUintField(&output, IREE_SV(".size"), 8);
  AppendStringField(
      &output, IREE_SV(".value_kind"),
      unknown_value_kind ? IREE_SV("made_up_kind") : IREE_SV("global_buffer"));
  AppendStringField(&output, IREE_SV(".address_space"), IREE_SV("global"));
  AppendStringField(&output, IREE_SV(".access"), IREE_SV("read_write"));
  AppendStringField(&output, IREE_SV(".actual_access"), IREE_SV("read_only"));
  AppendUintField(&output, IREE_SV(".align"), 8);

  AppendMsgPackMap(&output, 6);
  AppendUintField(&output, IREE_SV(".offset"), 8);
  AppendUintField(&output, IREE_SV(".size"), 8);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("global_buffer"));
  AppendStringField(&output, IREE_SV(".address_space"), IREE_SV("global"));
  AppendStringField(&output, IREE_SV(".access"), IREE_SV("write_only"));
  AppendUintField(&output, IREE_SV(".align"), 8);

  AppendMsgPackMap(&output, 4);
  AppendUintField(&output, IREE_SV(".offset"), 16);
  AppendUintField(&output, IREE_SV(".size"), 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  AppendUintField(&output, IREE_SV(".align"), 4);

  AppendMsgPackMap(&output, 4);
  AppendUintField(&output, IREE_SV(".offset"), out_of_range_arg ? 20 : 20);
  AppendUintField(&output, IREE_SV(".size"), out_of_range_arg ? 8 : 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  AppendUintField(&output, IREE_SV(".align"), 4);

  return output;
}

static std::vector<uint8_t> BuildMalformedMissingKernelFieldsMetadata() {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 1);
  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 0);
  return output;
}

static std::vector<uint8_t> BuildDuplicateArgumentFieldMetadata() {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 1);
  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 6);
  AppendStringField(&output, IREE_SV(".symbol"), IREE_SV("duplicate.kd"));
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 8);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 0);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 0);
  AppendMsgPackString(&output, IREE_SV(".args"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 4);
  AppendUintField(&output, IREE_SV(".offset"), 0);
  AppendUintField(&output, IREE_SV(".offset"), 4);
  AppendUintField(&output, IREE_SV(".size"), 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  return output;
}

static std::vector<uint8_t> BuildElfWithNote(
    const std::vector<uint8_t>& metadata, iree_string_view_t note_name,
    uint32_t note_type) {
  constexpr size_t kElfHeaderSize = 64;
  constexpr size_t kProgramHeaderOffset = 64;
  constexpr size_t kProgramHeaderSize = 56;
  constexpr size_t kNoteOffset = 128;

  std::vector<uint8_t> note;
  AppendU32LE(&note, (uint32_t)note_name.size + 1);
  AppendU32LE(&note, (uint32_t)metadata.size());
  AppendU32LE(&note, note_type);
  note.insert(note.end(), note_name.data, note_name.data + note_name.size);
  note.push_back(0);
  AppendAligned4Padding(&note);
  note.insert(note.end(), metadata.begin(), metadata.end());
  AppendAligned4Padding(&note);

  std::vector<uint8_t> elf(kNoteOffset, 0);
  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;               // ELFCLASS64.
  elf[5] = 1;               // ELFDATA2LSB.
  elf[6] = 1;               // EV_CURRENT.
  StoreU16LE(&elf, 16, 3);  // ET_DYN.
  StoreU16LE(&elf, 18, 224);
  StoreU32LE(&elf, 20, 1);
  StoreU64LE(&elf, 32, kProgramHeaderOffset);
  StoreU16LE(&elf, 52, kElfHeaderSize);
  StoreU16LE(&elf, 54, kProgramHeaderSize);
  StoreU16LE(&elf, 56, 1);

  StoreU32LE(&elf, kProgramHeaderOffset + 0, 4);  // PT_NOTE.
  StoreU64LE(&elf, kProgramHeaderOffset + 8, kNoteOffset);
  StoreU64LE(&elf, kProgramHeaderOffset + 32, note.size());
  StoreU64LE(&elf, kProgramHeaderOffset + 40, note.size());
  StoreU64LE(&elf, kProgramHeaderOffset + 48, 4);

  elf.insert(elf.end(), note.begin(), note.end());
  return elf;
}

static std::vector<uint8_t> BuildElfWithMetadata(
    const std::vector<uint8_t>& metadata) {
  return BuildElfWithNote(metadata, IREE_SV("AMDGPU"), 32);
}

static iree_const_byte_span_t ByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(AmdgpuMetadataTest, ParsesValidMetadata) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata());

  iree_hal_amdgpu_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  ASSERT_EQ(metadata.kernel_count, 1);
  ASSERT_EQ(metadata.arg_count, 4);
  ASSERT_GT(metadata.message_pack_data.data_length, 0);
  ASSERT_NE(metadata.kernels, nullptr);
  ASSERT_NE(metadata.args, nullptr);

  const iree_hal_amdgpu_metadata_kernel_t& kernel = metadata.kernels[0];
  EXPECT_EQ(ToString(kernel.name), "vector_add");
  EXPECT_EQ(ToString(kernel.symbol_name), "vector_add.kd");
  EXPECT_EQ(kernel.kernarg_segment_size, 24);
  EXPECT_EQ(kernel.kernarg_segment_alignment, 8);
  EXPECT_EQ(kernel.group_segment_fixed_size, 1024);
  EXPECT_EQ(kernel.private_segment_fixed_size, 64);
  ASSERT_TRUE(kernel.has_required_workgroup_size);
  EXPECT_EQ(kernel.required_workgroup_size[0], 16);
  EXPECT_EQ(kernel.required_workgroup_size[1], 4);
  EXPECT_EQ(kernel.required_workgroup_size[2], 1);
  ASSERT_EQ(kernel.arg_count, 4);
  ASSERT_EQ(kernel.args, metadata.args);

  EXPECT_EQ(kernel.args[0].offset, 0);
  EXPECT_EQ(kernel.args[0].size, 8);
  EXPECT_EQ(kernel.args[0].alignment, 8);
  EXPECT_EQ(kernel.args[0].kind,
            IREE_HAL_AMDGPU_METADATA_ARG_KIND_GLOBAL_BUFFER);
  EXPECT_EQ(ToString(kernel.args[0].value_kind), "global_buffer");
  EXPECT_EQ(ToString(kernel.args[0].address_space), "global");
  EXPECT_EQ(ToString(kernel.args[0].access), "read_only");

  EXPECT_EQ(kernel.args[1].offset, 8);
  EXPECT_EQ(kernel.args[1].size, 8);
  EXPECT_EQ(ToString(kernel.args[1].access), "write_only");

  EXPECT_EQ(kernel.args[2].offset, 16);
  EXPECT_EQ(kernel.args[2].size, 4);
  EXPECT_EQ(kernel.args[2].kind, IREE_HAL_AMDGPU_METADATA_ARG_KIND_BY_VALUE);

  EXPECT_EQ(kernel.args[3].offset, 20);
  EXPECT_EQ(kernel.args[3].size, 4);
  EXPECT_EQ(kernel.args[3].kind, IREE_HAL_AMDGPU_METADATA_ARG_KIND_BY_VALUE);

  iree_hal_amdgpu_metadata_deinitialize(&metadata);
}

TEST(AmdgpuMetadataTest, FindsKernelBySymbol) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata());

  iree_hal_amdgpu_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  const iree_hal_amdgpu_metadata_kernel_t* kernel = nullptr;
  IREE_EXPECT_OK(iree_hal_amdgpu_metadata_find_kernel_by_symbol(
      &metadata, IREE_SV("vector_add.kd"), &kernel));
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(ToString(kernel->name), "vector_add");

  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_metadata_find_kernel_by_symbol(
                            &metadata, IREE_SV("missing.kd"), &kernel));

  iree_hal_amdgpu_metadata_deinitialize(&metadata);
}

TEST(AmdgpuMetadataTest, AllowsUnknownValueKindAsOpaqueMetadata) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildKernelMetadata(/*out_of_range_arg=*/false,
                                               /*unknown_value_kind=*/true));

  iree_hal_amdgpu_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  ASSERT_EQ(metadata.kernel_count, 1);
  ASSERT_EQ(metadata.kernels[0].arg_count, 4);
  EXPECT_EQ(metadata.kernels[0].args[0].kind,
            IREE_HAL_AMDGPU_METADATA_ARG_KIND_UNKNOWN);
  EXPECT_EQ(ToString(metadata.kernels[0].args[0].value_kind), "made_up_kind");

  iree_hal_amdgpu_metadata_deinitialize(&metadata);
}

TEST(AmdgpuMetadataTest, RejectsOutOfRangeArgument) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildKernelMetadata(/*out_of_range_arg=*/true));

  iree_hal_amdgpu_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(AmdgpuMetadataTest, RejectsMissingMetadataNote) {
  std::vector<uint8_t> elf =
      BuildElfWithNote(BuildKernelMetadata(), IREE_SV("OTHER"), 32);

  iree_hal_amdgpu_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(AmdgpuMetadataTest, RejectsMalformedMessagePackMetadata) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildMalformedMissingKernelFieldsMetadata());

  iree_hal_amdgpu_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(AmdgpuMetadataTest, RejectsDuplicateArgumentField) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildDuplicateArgumentFieldMetadata());

  iree_hal_amdgpu_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(AmdgpuMetadataTest, TruncatedElfPrefixesNeverSucceed) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata());
  for (size_t length = 0; length < elf.size(); ++length) {
    iree_hal_amdgpu_metadata_t metadata;
    iree_status_t status = iree_hal_amdgpu_metadata_initialize_from_elf(
        iree_make_const_byte_span(elf.data(), length), iree_allocator_system(),
        &metadata);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_metadata_deinitialize(&metadata);
      ADD_FAILURE() << "unexpected success for truncated ELF prefix " << length;
      return;
    }
    iree_status_ignore(status);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
