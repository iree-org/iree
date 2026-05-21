// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

static void iree_hal_amdgpu_hsaco_metadata_fuzz_append_u32le(
    std::vector<uint8_t>* output, uint32_t value) {
  output->push_back((uint8_t)value);
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)(value >> 16));
  output->push_back((uint8_t)(value >> 24));
}

static void iree_hal_amdgpu_hsaco_metadata_fuzz_store_u16le(
    std::vector<uint8_t>* output, size_t offset, uint16_t value) {
  (*output)[offset + 0] = (uint8_t)value;
  (*output)[offset + 1] = (uint8_t)(value >> 8);
}

static void iree_hal_amdgpu_hsaco_metadata_fuzz_store_u32le(
    std::vector<uint8_t>* output, size_t offset, uint32_t value) {
  (*output)[offset + 0] = (uint8_t)value;
  (*output)[offset + 1] = (uint8_t)(value >> 8);
  (*output)[offset + 2] = (uint8_t)(value >> 16);
  (*output)[offset + 3] = (uint8_t)(value >> 24);
}

static void iree_hal_amdgpu_hsaco_metadata_fuzz_store_u64le(
    std::vector<uint8_t>* output, size_t offset, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    (*output)[offset + i] = (uint8_t)(value >> (i * 8));
  }
}

static void iree_hal_amdgpu_hsaco_metadata_fuzz_append_aligned4_padding(
    std::vector<uint8_t>* output) {
  while ((output->size() & 3) != 0) output->push_back(0);
}

static std::vector<uint8_t> iree_hal_amdgpu_hsaco_metadata_fuzz_wrap_as_elf(
    const uint8_t* data, size_t size) {
  constexpr size_t kElfHeaderSize = 64;
  constexpr size_t kProgramHeaderOffset = 64;
  constexpr size_t kProgramHeaderSize = 56;
  constexpr size_t kNoteOffset = 128;
  static const uint8_t kNoteName[] = {'A', 'M', 'D', 'G', 'P', 'U'};

  std::vector<uint8_t> note;
  iree_hal_amdgpu_hsaco_metadata_fuzz_append_u32le(&note,
                                                   sizeof(kNoteName) + 1);
  iree_hal_amdgpu_hsaco_metadata_fuzz_append_u32le(&note, (uint32_t)size);
  iree_hal_amdgpu_hsaco_metadata_fuzz_append_u32le(&note, 32);
  note.insert(note.end(), kNoteName, kNoteName + sizeof(kNoteName));
  note.push_back(0);
  iree_hal_amdgpu_hsaco_metadata_fuzz_append_aligned4_padding(&note);
  note.insert(note.end(), data, data + size);
  iree_hal_amdgpu_hsaco_metadata_fuzz_append_aligned4_padding(&note);

  std::vector<uint8_t> elf(kNoteOffset, 0);
  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;
  elf[5] = 1;
  elf[6] = 1;
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u16le(&elf, 16, 3);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u16le(&elf, 18, 224);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u32le(&elf, 20, 1);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u64le(&elf, 32,
                                                  kProgramHeaderOffset);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u16le(&elf, 52, kElfHeaderSize);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u16le(&elf, 54, kProgramHeaderSize);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u16le(&elf, 56, 1);

  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u32le(&elf, kProgramHeaderOffset,
                                                  4);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u64le(
      &elf, kProgramHeaderOffset + 8, kNoteOffset);
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u64le(
      &elf, kProgramHeaderOffset + 32, note.size());
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u64le(
      &elf, kProgramHeaderOffset + 40, note.size());
  iree_hal_amdgpu_hsaco_metadata_fuzz_store_u64le(&elf,
                                                  kProgramHeaderOffset + 48, 4);

  elf.insert(elf.end(), note.begin(), note.end());
  return elf;
}

static void iree_hal_amdgpu_hsaco_metadata_fuzz_parse(
    iree_const_byte_span_t elf_data) {
  iree_hal_amdgpu_hsaco_metadata_t metadata;
  iree_status_t status = iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      elf_data, iree_allocator_system(), &metadata);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
  } else {
    iree_status_free(status);
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  constexpr size_t kMaxInputSize = 64 * 1024;
  if (size > kMaxInputSize) size = kMaxInputSize;

  iree_hal_amdgpu_hsaco_metadata_fuzz_parse(
      iree_make_const_byte_span(data, size));

  std::vector<uint8_t> elf =
      iree_hal_amdgpu_hsaco_metadata_fuzz_wrap_as_elf(data, size);
  iree_hal_amdgpu_hsaco_metadata_fuzz_parse(
      iree_make_const_byte_span(elf.data(), elf.size()));
  return 0;
}
