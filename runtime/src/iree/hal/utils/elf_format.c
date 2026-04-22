// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/elf_format.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// ELF Helpers
//===----------------------------------------------------------------------===//

// ELF magic bytes: 0x7F 'E' 'L' 'F'.
static const uint8_t iree_hal_elf_magic[4] = {0x7F, 'E', 'L', 'F'};

typedef enum iree_hal_elf_ident_e {
  IREE_HAL_ELF_EI_CLASS = 4,
  IREE_HAL_ELF_EI_DATA = 5,
} iree_hal_elf_ident_t;

typedef enum iree_hal_elf_class_e {
  IREE_HAL_ELF_CLASS_32 = 1,
  IREE_HAL_ELF_CLASS_64 = 2,
} iree_hal_elf_class_t;

typedef enum iree_hal_elf_data_e {
  IREE_HAL_ELF_DATA_2LSB = 1,
} iree_hal_elf_data_t;

typedef struct iree_hal_elf_ehdr_common_t {
  uint8_t e_ident[16];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
} iree_hal_elf_ehdr_common_t;

typedef struct iree_hal_elf32_ehdr_t {
  iree_hal_elf_ehdr_common_t common;
  uint32_t e_entry;
  uint32_t e_phoff;
  uint32_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
} iree_hal_elf32_ehdr_t;

typedef struct iree_hal_elf64_ehdr_t {
  iree_hal_elf_ehdr_common_t common;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
} iree_hal_elf64_ehdr_t;

typedef struct iree_hal_elf32_shdr_t {
  uint32_t sh_name;
  uint32_t sh_type;
  uint32_t sh_flags;
  uint32_t sh_addr;
  uint32_t sh_offset;
  uint32_t sh_size;
} iree_hal_elf32_shdr_t;

typedef struct iree_hal_elf64_shdr_t {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
} iree_hal_elf64_shdr_t;

bool iree_hal_elf_data_starts_with_magic(iree_const_byte_span_t elf_data) {
  if (elf_data.data_length != 0 &&
      elf_data.data_length < sizeof(iree_hal_elf_magic)) {
    return false;
  }
  iree_const_byte_span_t magic_data =
      iree_make_const_byte_span(elf_data.data, sizeof(iree_hal_elf_magic));
  if (iree_const_byte_span_is_empty(magic_data)) return false;
  return memcmp(magic_data.data, iree_hal_elf_magic, magic_data.data_length) ==
         0;
}

static bool iree_hal_elf_has_available_bytes(iree_const_byte_span_t elf_data,
                                             uint64_t byte_count) {
  return elf_data.data_length == 0 || byte_count <= elf_data.data_length;
}

static iree_status_t iree_hal_elf_calculate_table_end(uint64_t offset,
                                                      uint64_t count,
                                                      uint64_t element_size,
                                                      uint64_t* out_end) {
  if (count != 0 && element_size > UINT64_MAX / count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF table size overflow");
  }
  const uint64_t table_size = count * element_size;
  if (offset > UINT64_MAX - table_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF table extent overflow");
  }
  *out_end = offset + table_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_elf_calculate_size_32(
    const iree_hal_elf32_ehdr_t* header, iree_const_byte_span_t elf_data,
    uint64_t* inout_max_offset) {
  uint64_t max_offset = *inout_max_offset;
  if (header->e_phoff != 0) {
    uint64_t ph_end = 0;
    IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_table_end(
        header->e_phoff, header->e_phnum, header->e_phentsize, &ph_end));
    max_offset = iree_max(max_offset, ph_end);
  }
  if (header->e_shoff != 0) {
    if (header->e_shnum != 0 &&
        header->e_shentsize < sizeof(iree_hal_elf32_shdr_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "32-bit ELF section header entry too small");
    }
    uint64_t sh_table_end = 0;
    IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_table_end(
        header->e_shoff, header->e_shnum, header->e_shentsize, &sh_table_end));
    max_offset = iree_max(max_offset, sh_table_end);
    if (!iree_hal_elf_has_available_bytes(elf_data, sh_table_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF section header table is truncated");
    }
    const uint8_t* sh_table = elf_data.data + header->e_shoff;
    for (uint16_t i = 0; i < header->e_shnum; ++i) {
      const iree_hal_elf32_shdr_t* section_header =
          (const iree_hal_elf32_shdr_t*)(sh_table + i * header->e_shentsize);
      uint64_t section_end = 0;
      IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_table_end(
          section_header->sh_offset, /*count=*/1, section_header->sh_size,
          &section_end));
      max_offset = iree_max(max_offset, section_end);
    }
  }
  *inout_max_offset = max_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_elf_calculate_size_64(
    const iree_hal_elf64_ehdr_t* header, iree_const_byte_span_t elf_data,
    uint64_t* inout_max_offset) {
  uint64_t max_offset = *inout_max_offset;
  if (header->e_phoff != 0) {
    uint64_t ph_end = 0;
    IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_table_end(
        header->e_phoff, header->e_phnum, header->e_phentsize, &ph_end));
    max_offset = iree_max(max_offset, ph_end);
  }
  if (header->e_shoff != 0) {
    if (header->e_shnum != 0 &&
        header->e_shentsize < sizeof(iree_hal_elf64_shdr_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "64-bit ELF section header entry too small");
    }
    uint64_t sh_table_end = 0;
    IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_table_end(
        header->e_shoff, header->e_shnum, header->e_shentsize, &sh_table_end));
    max_offset = iree_max(max_offset, sh_table_end);
    if (!iree_hal_elf_has_available_bytes(elf_data, sh_table_end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF section header table is truncated");
    }
    const uint8_t* sh_table = elf_data.data + header->e_shoff;
    for (uint16_t i = 0; i < header->e_shnum; ++i) {
      const iree_hal_elf64_shdr_t* section_header =
          (const iree_hal_elf64_shdr_t*)(sh_table + i * header->e_shentsize);
      uint64_t section_end = 0;
      IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_table_end(
          section_header->sh_offset, /*count=*/1, section_header->sh_size,
          &section_end));
      max_offset = iree_max(max_offset, section_end);
    }
  }
  *inout_max_offset = max_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_elf_calculate_size(iree_const_byte_span_t elf_data,
                                          iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;
  if (!iree_hal_elf_data_starts_with_magic(elf_data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "data does not begin with ELF magic");
  }
  if (!iree_hal_elf_has_available_bytes(elf_data,
                                        sizeof(iree_hal_elf_ehdr_common_t))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "ELF header truncated");
  }

  const iree_hal_elf_ehdr_common_t* common_header =
      (const iree_hal_elf_ehdr_common_t*)elf_data.data;
  if (common_header->e_ident[IREE_HAL_ELF_EI_DATA] != IREE_HAL_ELF_DATA_2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only little-endian ELF files are supported");
  }

  const uint8_t elf_class = common_header->e_ident[IREE_HAL_ELF_EI_CLASS];
  uint64_t max_offset = 0;
  if (elf_class == IREE_HAL_ELF_CLASS_64) {
    if (!iree_hal_elf_has_available_bytes(elf_data,
                                          sizeof(iree_hal_elf64_ehdr_t))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "64-bit ELF header truncated");
    }
    max_offset = sizeof(iree_hal_elf64_ehdr_t);
    IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_size_64(
        (const iree_hal_elf64_ehdr_t*)elf_data.data, elf_data, &max_offset));
  } else if (elf_class == IREE_HAL_ELF_CLASS_32) {
    if (!iree_hal_elf_has_available_bytes(elf_data,
                                          sizeof(iree_hal_elf32_ehdr_t))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "32-bit ELF header truncated");
    }
    max_offset = sizeof(iree_hal_elf32_ehdr_t);
    IREE_RETURN_IF_ERROR(iree_hal_elf_calculate_size_32(
        (const iree_hal_elf32_ehdr_t*)elf_data.data, elf_data, &max_offset));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ELF class %u", elf_class);
  }

  if (max_offset > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF size exceeds host size range");
  }
  *out_size = (iree_host_size_t)max_offset;
  return iree_ok_status();
}
