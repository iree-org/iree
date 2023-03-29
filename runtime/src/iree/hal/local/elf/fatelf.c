// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/elf/fatelf.h"

#include "iree/base/target_platform.h"
#include "iree/hal/local/elf/arch.h"

iree_status_t iree_fatelf_select(iree_const_byte_span_t file_data,
                                 iree_const_byte_span_t* out_elf_data) {
  *out_elf_data = iree_const_byte_span_empty();

  // If there's not enough room for the header and a single record then don't
  // bother checking.
  if (file_data.data_length <
      sizeof(iree_fatelf_header_t) + sizeof(iree_fatelf_record_t)) {
    *out_elf_data = file_data;
    return iree_ok_status();
  }

  // NOTE: header is always in little-endian so we have to use our utilities to
  // be portable. We don't actually care about big-endian platforms but this
  // code may end up there someday.
  const iree_fatelf_header_t* raw_header =
      (const iree_fatelf_header_t*)file_data.data;
  iree_fatelf_header_t host_header = {
      .magic = iree_unaligned_load_le_u32(&raw_header->magic),
      .version = iree_unaligned_load_le_u16(&raw_header->version),
      .record_count = iree_unaligned_load_le_u8(&raw_header->record_count),
      .reserved = iree_unaligned_load_le_u8(&raw_header->reserved),
  };

  // Ignore if not a FatELF.
  // After this point we have to fail if there are issues as what's calling us
  // will not be able to do anything with the file if it's a FatELF.
  if (host_header.magic != IREE_FATELF_MAGIC) {
    *out_elf_data = file_data;
    return iree_ok_status();
  }
  if (host_header.version != IREE_FATELF_FORMAT_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FatELF has version %d but runtime only supports version %d",
        host_header.version, IREE_FATELF_FORMAT_VERSION);
  }

  // Ensure there's enough space for all the declared records.
  iree_host_size_t required_bytes =
      sizeof(iree_fatelf_header_t) +
      host_header.record_count * sizeof(iree_fatelf_record_t);
  if (file_data.data_length < required_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FatELF file truncated, requires at least %" PRIhsz
                            "B for headers but only have %" PRIhsz
                            "B available",
                            required_bytes, file_data.data_length);
  }

  // Scan record table to find one that matches.
  iree_elf64_off_t selected_offset = 0;
  iree_elf64_xword_t selected_size = 0;
  for (iree_elf64_byte_t i = 0; i < host_header.record_count; ++i) {
    const iree_fatelf_record_t* raw_record = &raw_header->records[i];
    const iree_fatelf_record_t host_record = {
        .machine = iree_unaligned_load_le_u16(&raw_record->machine),
        .osabi = iree_unaligned_load_le_u8(&raw_record->osabi),
        .osabi_version = iree_unaligned_load_le_u8(&raw_record->osabi_version),
        .word_size = iree_unaligned_load_le_u8(&raw_record->word_size),
        .byte_order = iree_unaligned_load_le_u8(&raw_record->byte_order),
        .reserved0 = iree_unaligned_load_le_u8(&raw_record->reserved0),
        .reserved1 = iree_unaligned_load_le_u8(&raw_record->reserved1),
        .offset = iree_unaligned_load_le_u64(&raw_record->offset),
        .size = iree_unaligned_load_le_u64(&raw_record->size),
    };
    if (!iree_elf_machine_is_valid(host_record.machine)) continue;
    if (host_record.osabi != IREE_ELF_ELFOSABI_NONE &&
        host_record.osabi != IREE_ELF_ELFOSABI_LINUX &&
        host_record.osabi != IREE_ELF_ELFOSABI_STANDALONE) {
      // We're standalone but follow the Linux ABI.
      continue;
    }
#if defined(IREE_PTR_SIZE_32)
    if (host_record.word_size != IREE_FATELF_WORD_SIZE_32) continue;
#else
    if (host_record.word_size != IREE_FATELF_WORD_SIZE_64) continue;
#endif  // IREE_PTR_SIZE_32
#if IREE_ENDIANNESS_LITTLE
    if (host_record.byte_order != IREE_FATELF_BYTE_ORDER_LSB) continue;
#else
    if (host_record.byte_order != IREE_FATELF_BYTE_ORDER_MSB) continue;
#endif  // IREE_ENDIANNESS_LITTLE
    selected_offset = host_record.offset;
    selected_size = host_record.size;
    break;
  }
  if (!selected_offset || !selected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no ELFs matching the runtime architecture or "
                            "Linux ABI found in the FatELF");
  }

  // Bounds check the file range - the caller expects valid pointers.
  if (selected_offset < required_bytes ||
      selected_offset + selected_size > file_data.data_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "ELF file range out of bounds; %" PRIu64 "-%" PRIu64 " (%" PRIu64
        ") specified out of %" PRIhsz " valid bytes",
        selected_offset, selected_offset + selected_size - 1, selected_size,
        file_data.data_length);
  }

  *out_elf_data = iree_make_const_byte_span(file_data.data + selected_offset,
                                            selected_size);
  return iree_ok_status();
}
