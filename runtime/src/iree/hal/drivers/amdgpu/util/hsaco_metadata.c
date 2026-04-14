// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// ELF note discovery
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_ELF_MAGIC0 0x7F
#define IREE_HAL_AMDGPU_ELF_MAGIC1 'E'
#define IREE_HAL_AMDGPU_ELF_MAGIC2 'L'
#define IREE_HAL_AMDGPU_ELF_MAGIC3 'F'
#define IREE_HAL_AMDGPU_ELF_CLASS_64 2
#define IREE_HAL_AMDGPU_ELF_DATA_LITTLE 1
#define IREE_HAL_AMDGPU_ELF_VERSION_CURRENT 1
#define IREE_HAL_AMDGPU_ELF_MACHINE_AMDGPU 224
#define IREE_HAL_AMDGPU_ELF_PT_NOTE 4
#define IREE_HAL_AMDGPU_ELF_NOTE_AMDGPU_METADATA 32

#define IREE_HAL_AMDGPU_ELF64_HEADER_SIZE 64
#define IREE_HAL_AMDGPU_ELF64_PROGRAM_HEADER_SIZE 56

static uint16_t iree_hal_amdgpu_hsaco_metadata_load_le_u16(const uint8_t* ptr) {
  return iree_unaligned_load_le((const uint16_t*)ptr);
}

static uint32_t iree_hal_amdgpu_hsaco_metadata_load_le_u32(const uint8_t* ptr) {
  return iree_unaligned_load_le((const uint32_t*)ptr);
}

static uint64_t iree_hal_amdgpu_hsaco_metadata_load_le_u64(const uint8_t* ptr) {
  return iree_unaligned_load_le((const uint64_t*)ptr);
}

static bool iree_hal_amdgpu_hsaco_metadata_range_in_bounds(
    iree_host_size_t offset, iree_host_size_t length, iree_host_size_t limit) {
  iree_host_size_t end = 0;
  return offset <= limit && iree_host_size_checked_add(offset, length, &end) &&
         end <= limit;
}

static bool iree_hal_amdgpu_hsaco_metadata_u64_to_host_size(
    uint64_t value, iree_host_size_t* out_value) {
  if (value > (uint64_t)IREE_HOST_SIZE_MAX) return false;
  *out_value = (iree_host_size_t)value;
  return true;
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_checked_align4(
    iree_host_size_t value, iree_host_size_t* out_aligned_value) {
  if (!iree_host_size_checked_align(value, 4, out_aligned_value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ELF note alignment overflow");
  }
  return iree_ok_status();
}

static iree_string_view_t iree_hal_amdgpu_hsaco_metadata_note_name_view(
    const uint8_t* data, iree_host_size_t length) {
  if (length > 0 && data[length - 1] == 0) --length;
  return iree_make_string_view((const char*)data, length);
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_scan_note_segment(
    iree_const_byte_span_t segment_data,
    iree_const_byte_span_t* out_message_pack_data, bool* out_found) {
  *out_found = false;
  iree_host_size_t offset = 0;
  while (segment_data.data_length - offset >= 12) {
    const uint8_t* note_header = segment_data.data + offset;
    const uint32_t name_size =
        iree_hal_amdgpu_hsaco_metadata_load_le_u32(note_header + 0);
    const uint32_t desc_size =
        iree_hal_amdgpu_hsaco_metadata_load_le_u32(note_header + 4);
    const uint32_t note_type =
        iree_hal_amdgpu_hsaco_metadata_load_le_u32(note_header + 8);
    offset += 12;

    const iree_host_size_t name_offset = offset;
    if (!iree_hal_amdgpu_hsaco_metadata_range_in_bounds(
            name_offset, name_size, segment_data.data_length)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU ELF note name exceeds PT_NOTE bounds");
    }
    iree_host_size_t desc_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_checked_align4(
        name_offset + name_size, &desc_offset));
    if (desc_offset > segment_data.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU ELF note descriptor offset exceeds "
                              "PT_NOTE bounds");
    }
    if (!iree_hal_amdgpu_hsaco_metadata_range_in_bounds(
            desc_offset, desc_size, segment_data.data_length)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU ELF note descriptor exceeds PT_NOTE "
                              "bounds");
    }
    iree_host_size_t next_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_checked_align4(
        desc_offset + desc_size, &next_offset));
    if (next_offset > segment_data.data_length) {
      // Some producers omit final padding from the segment size. The descriptor
      // itself is still fully present, so allow the final record to end exactly
      // at the segment end.
      if (desc_offset + desc_size == segment_data.data_length) {
        next_offset = segment_data.data_length;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU ELF note padding exceeds PT_NOTE "
                                "bounds");
      }
    }

    iree_string_view_t note_name =
        iree_hal_amdgpu_hsaco_metadata_note_name_view(
            segment_data.data + name_offset, name_size);
    if (note_type == IREE_HAL_AMDGPU_ELF_NOTE_AMDGPU_METADATA &&
        iree_string_view_equal(note_name, IREE_SV("AMDGPU"))) {
      *out_message_pack_data =
          iree_make_const_byte_span(segment_data.data + desc_offset, desc_size);
      *out_found = true;
      return iree_ok_status();
    }

    offset = next_offset;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_find_note(
    iree_const_byte_span_t elf_data,
    iree_const_byte_span_t* out_message_pack_data) {
  *out_message_pack_data = iree_const_byte_span_empty();
  if (elf_data.data_length < IREE_HAL_AMDGPU_ELF64_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ELF data too small");
  }
  const uint8_t* header = elf_data.data;
  if (header[0] != IREE_HAL_AMDGPU_ELF_MAGIC0 ||
      header[1] != IREE_HAL_AMDGPU_ELF_MAGIC1 ||
      header[2] != IREE_HAL_AMDGPU_ELF_MAGIC2 ||
      header[3] != IREE_HAL_AMDGPU_ELF_MAGIC3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata input is not an ELF file");
  }
  if (header[4] != IREE_HAL_AMDGPU_ELF_CLASS_64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata ELF must be 64-bit");
  }
  if (header[5] != IREE_HAL_AMDGPU_ELF_DATA_LITTLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata ELF must be little-endian");
  }
  if (header[6] != IREE_HAL_AMDGPU_ELF_VERSION_CURRENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata ELF has unsupported version");
  }
  const uint16_t machine =
      iree_hal_amdgpu_hsaco_metadata_load_le_u16(header + 18);
  if (machine != IREE_HAL_AMDGPU_ELF_MACHINE_AMDGPU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata ELF has non-AMDGPU machine %u",
                            machine);
  }

  iree_host_size_t program_header_offset = 0;
  if (!iree_hal_amdgpu_hsaco_metadata_u64_to_host_size(
          iree_hal_amdgpu_hsaco_metadata_load_le_u64(header + 32),
          &program_header_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ELF program header offset overflows host "
                            "size");
  }
  const uint16_t program_header_entry_size =
      iree_hal_amdgpu_hsaco_metadata_load_le_u16(header + 54);
  const uint16_t program_header_count =
      iree_hal_amdgpu_hsaco_metadata_load_le_u16(header + 56);
  if (program_header_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU ELF has no program headers");
  }
  if (program_header_entry_size < IREE_HAL_AMDGPU_ELF64_PROGRAM_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ELF program header entries are too small");
  }
  iree_host_size_t program_headers_size = 0;
  if (!iree_host_size_checked_mul(program_header_count,
                                  program_header_entry_size,
                                  &program_headers_size) ||
      !iree_hal_amdgpu_hsaco_metadata_range_in_bounds(
          program_header_offset, program_headers_size, elf_data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ELF program headers exceed file bounds");
  }

  for (uint16_t i = 0; i < program_header_count; ++i) {
    const uint8_t* program_header =
        elf_data.data + program_header_offset +
        (iree_host_size_t)i * program_header_entry_size;
    const uint32_t program_header_type =
        iree_hal_amdgpu_hsaco_metadata_load_le_u32(program_header + 0);
    if (program_header_type != IREE_HAL_AMDGPU_ELF_PT_NOTE) continue;

    iree_host_size_t note_offset = 0;
    iree_host_size_t note_size = 0;
    if (!iree_hal_amdgpu_hsaco_metadata_u64_to_host_size(
            iree_hal_amdgpu_hsaco_metadata_load_le_u64(program_header + 8),
            &note_offset) ||
        !iree_hal_amdgpu_hsaco_metadata_u64_to_host_size(
            iree_hal_amdgpu_hsaco_metadata_load_le_u64(program_header + 32),
            &note_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU ELF PT_NOTE range overflows host size");
    }
    if (!iree_hal_amdgpu_hsaco_metadata_range_in_bounds(note_offset, note_size,
                                                        elf_data.data_length)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU ELF PT_NOTE exceeds file bounds");
    }
    bool found = false;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_scan_note_segment(
        iree_make_const_byte_span(elf_data.data + note_offset, note_size),
        out_message_pack_data, &found));
    if (found) return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "AMDGPU metadata note not found");
}

//===----------------------------------------------------------------------===//
// MessagePack reader
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_msgpack_reader_t {
  const uint8_t* current;
  const uint8_t* end;
} iree_hal_amdgpu_msgpack_reader_t;

static iree_host_size_t iree_hal_amdgpu_msgpack_remaining(
    const iree_hal_amdgpu_msgpack_reader_t* reader) {
  return (iree_host_size_t)(reader->end - reader->current);
}

static iree_status_t iree_hal_amdgpu_msgpack_require(
    const iree_hal_amdgpu_msgpack_reader_t* reader, iree_host_size_t length) {
  if (iree_hal_amdgpu_msgpack_remaining(reader) < length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "truncated AMDGPU MessagePack metadata");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_read_u8(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint8_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_require(reader, 1));
  *out_value = *reader->current++;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_read_be_u16(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint16_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_require(reader, 2));
  *out_value = ((uint16_t)reader->current[0] << 8) | reader->current[1];
  reader->current += 2;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_read_be_u32(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint32_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_require(reader, 4));
  *out_value = ((uint32_t)reader->current[0] << 24) |
               ((uint32_t)reader->current[1] << 16) |
               ((uint32_t)reader->current[2] << 8) | reader->current[3];
  reader->current += 4;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_read_be_u64(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint64_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_require(reader, 8));
  *out_value = ((uint64_t)reader->current[0] << 56) |
               ((uint64_t)reader->current[1] << 48) |
               ((uint64_t)reader->current[2] << 40) |
               ((uint64_t)reader->current[3] << 32) |
               ((uint64_t)reader->current[4] << 24) |
               ((uint64_t)reader->current[5] << 16) |
               ((uint64_t)reader->current[6] << 8) | reader->current[7];
  reader->current += 8;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_skip_bytes(
    iree_hal_amdgpu_msgpack_reader_t* reader, iree_host_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_require(reader, length));
  reader->current += length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_read_count_after_tag(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint8_t tag, uint8_t fix_base,
    uint8_t type16, uint8_t type32, uint32_t* out_count) {
  if ((tag & 0xF0u) == fix_base) {
    *out_count = tag & 0x0Fu;
    return iree_ok_status();
  }
  if (tag == type16) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u16(reader, &value));
    *out_count = value;
    return iree_ok_status();
  }
  if (tag == type32) {
    return iree_hal_amdgpu_msgpack_read_be_u32(reader, out_count);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unexpected AMDGPU MessagePack container tag 0x%02X",
                          tag);
}

static iree_status_t iree_hal_amdgpu_msgpack_read_map_count(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint32_t* out_count) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &tag));
  return iree_hal_amdgpu_msgpack_read_count_after_tag(reader, tag, 0x80, 0xDE,
                                                      0xDF, out_count);
}

static iree_status_t iree_hal_amdgpu_msgpack_read_array_count(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint32_t* out_count) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &tag));
  if ((tag & 0xF0u) == 0x90u) {
    *out_count = tag & 0x0Fu;
    return iree_ok_status();
  }
  if (tag == 0xDC) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u16(reader, &value));
    *out_count = value;
    return iree_ok_status();
  }
  if (tag == 0xDD) {
    return iree_hal_amdgpu_msgpack_read_be_u32(reader, out_count);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected AMDGPU MessagePack array tag, got 0x%02X",
                          tag);
}

static iree_status_t iree_hal_amdgpu_msgpack_read_string_after_tag(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint8_t tag,
    iree_string_view_t* out_value) {
  uint32_t length = 0;
  if ((tag & 0xE0u) == 0xA0u) {
    length = tag & 0x1Fu;
  } else if (tag == 0xD9) {
    uint8_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &value));
    length = value;
  } else if (tag == 0xDA) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u16(reader, &value));
    length = value;
  } else if (tag == 0xDB) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u32(reader, &length));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected AMDGPU MessagePack string tag, got "
                            "0x%02X",
                            tag);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_require(reader, length));
  *out_value = iree_make_string_view((const char*)reader->current, length);
  reader->current += length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_read_string(
    iree_hal_amdgpu_msgpack_reader_t* reader, iree_string_view_t* out_value) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &tag));
  return iree_hal_amdgpu_msgpack_read_string_after_tag(reader, tag, out_value);
}

static iree_status_t iree_hal_amdgpu_msgpack_read_uint64(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint64_t* out_value) {
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &tag));
  if (tag <= 0x7F) {
    *out_value = tag;
    return iree_ok_status();
  }
  switch (tag) {
    case 0xCC: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &value));
      *out_value = value;
      return iree_ok_status();
    }
    case 0xCD: {
      uint16_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u16(reader, &value));
      *out_value = value;
      return iree_ok_status();
    }
    case 0xCE: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u32(reader, &value));
      *out_value = value;
      return iree_ok_status();
    }
    case 0xCF:
      return iree_hal_amdgpu_msgpack_read_be_u64(reader, out_value);
    case 0xD0: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &value));
      if (value & 0x80u) break;
      *out_value = value;
      return iree_ok_status();
    }
    case 0xD1: {
      uint16_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u16(reader, &value));
      if (value & 0x8000u) break;
      *out_value = value;
      return iree_ok_status();
    }
    case 0xD2: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u32(reader, &value));
      if (value & 0x80000000u) break;
      *out_value = value;
      return iree_ok_status();
    }
    case 0xD3: {
      uint64_t value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_be_u64(reader, &value));
      if (value & 0x8000000000000000ull) break;
      *out_value = value;
      return iree_ok_status();
    }
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected non-negative AMDGPU MessagePack integer");
}

static iree_status_t iree_hal_amdgpu_msgpack_read_uint32(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint32_t* out_value) {
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_uint64(reader, &value));
  if (value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata integer exceeds uint32_t");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_msgpack_skip(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint32_t depth) {
  if (depth > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU MessagePack metadata is nested too deeply");
  }
  uint8_t tag = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &tag));
  if (tag <= 0x7F || tag >= 0xE0 || tag == 0xC0 || tag == 0xC2 || tag == 0xC3) {
    return iree_ok_status();
  }
  if ((tag & 0xE0u) == 0xA0u) {
    return iree_hal_amdgpu_msgpack_skip_bytes(reader, tag & 0x1Fu);
  }
  if ((tag & 0xF0u) == 0x90u) {
    const uint32_t count = tag & 0x0Fu;
    for (uint32_t i = 0; i < count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, depth + 1));
    }
    return iree_ok_status();
  }
  if ((tag & 0xF0u) == 0x80u) {
    const uint32_t count = tag & 0x0Fu;
    for (uint32_t i = 0; i < count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, depth + 1));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, depth + 1));
    }
    return iree_ok_status();
  }

  switch (tag) {
    case 0xC4: {
      uint8_t length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &length));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC5: {
      uint16_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_be_u16(reader, &length));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC6: {
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_be_u32(reader, &length));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC7: {
      uint8_t length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &length));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip_bytes(reader, 1));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC8: {
      uint16_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_be_u16(reader, &length));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip_bytes(reader, 1));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xC9: {
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_be_u32(reader, &length));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip_bytes(reader, 1));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xCA:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 4);
    case 0xCB:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 8);
    case 0xCC:
    case 0xD0:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 1);
    case 0xCD:
    case 0xD1:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 2);
    case 0xCE:
    case 0xD2:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 4);
    case 0xCF:
    case 0xD3:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 8);
    case 0xD4:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 2);
    case 0xD5:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 3);
    case 0xD6:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 5);
    case 0xD7:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 9);
    case 0xD8:
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, 17);
    case 0xD9: {
      uint8_t length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_u8(reader, &length));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xDA: {
      uint16_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_be_u16(reader, &length));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xDB: {
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_be_u32(reader, &length));
      return iree_hal_amdgpu_msgpack_skip_bytes(reader, length);
    }
    case 0xDC:
    case 0xDD: {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_count_after_tag(
          reader, tag, 0x90, 0xDC, 0xDD, &count));
      for (uint32_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, depth + 1));
      }
      return iree_ok_status();
    }
    case 0xDE:
    case 0xDF: {
      uint32_t count = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_count_after_tag(
          reader, tag, 0x80, 0xDE, 0xDF, &count));
      for (uint32_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, depth + 1));
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, depth + 1));
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AMDGPU MessagePack tag 0x%02X", tag);
  }
}

//===----------------------------------------------------------------------===//
// AMDGPU metadata decoding
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_hsaco_metadata_count_t {
  iree_host_size_t kernel_count;
  iree_host_size_t arg_count;
} iree_hal_amdgpu_hsaco_metadata_count_t;

typedef struct iree_hal_amdgpu_hsaco_metadata_kernel_fields_t {
  bool has_name;
  bool has_symbol_name;
  bool has_kernarg_segment_size;
  bool has_kernarg_segment_alignment;
  bool has_group_segment_fixed_size;
  bool has_private_segment_fixed_size;
  bool has_required_workgroup_size;
  bool has_args;
} iree_hal_amdgpu_hsaco_metadata_kernel_fields_t;

typedef struct iree_hal_amdgpu_hsaco_metadata_arg_fields_t {
  bool has_offset;
  bool has_size;
  bool has_value_kind;
  bool has_address_space;
  bool has_access;
  bool has_actual_access;
  bool has_alignment;
} iree_hal_amdgpu_hsaco_metadata_arg_fields_t;

static iree_status_t iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(
    iree_string_view_t key) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU metadata repeats field `%.*s`", (int)key.size,
                          key.data);
}

static iree_hal_amdgpu_hsaco_metadata_arg_kind_t
iree_hal_amdgpu_hsaco_metadata_classify_arg_kind(
    iree_string_view_t value_kind) {
  if (iree_string_view_equal(value_kind, IREE_SV("by_value"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("global_buffer"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("dynamic_shared_pointer"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_DYNAMIC_SHARED_POINTER;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("image"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_IMAGE;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("sampler"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_SAMPLER;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("pipe"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_PIPE;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("queue"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_QUEUE;
  }
  if (iree_string_view_equal(value_kind, IREE_SV("hidden_none"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN_NONE;
  }
  if (iree_string_view_starts_with(value_kind, IREE_SV("hidden_"))) {
    return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN;
  }
  return IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_UNKNOWN;
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_count_kernel_args(
    iree_hal_amdgpu_msgpack_reader_t* reader,
    iree_host_size_t* inout_arg_count) {
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_msgpack_read_map_count(reader, &field_count));
  bool has_args = false;
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_string(reader, &key));
    if (iree_string_view_equal(key, IREE_SV(".args"))) {
      if (has_args) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU kernel metadata repeats `.args`");
      }
      has_args = true;
      uint32_t arg_count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_array_count(reader, &arg_count));
      if (!iree_host_size_checked_add(*inout_arg_count, arg_count,
                                      inout_arg_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU metadata argument count overflow");
      }
      for (uint32_t j = 0; j < arg_count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, 0));
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, 0));
    }
  }
  if (!has_args) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel metadata missing `.args`");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_count_message_pack(
    iree_const_byte_span_t message_pack_data,
    iree_hal_amdgpu_hsaco_metadata_count_t* out_count) {
  memset(out_count, 0, sizeof(*out_count));
  iree_hal_amdgpu_msgpack_reader_t reader = {
      .current = message_pack_data.data,
      .end = message_pack_data.data + message_pack_data.data_length,
  };
  uint32_t root_field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_msgpack_read_map_count(&reader, &root_field_count));
  bool has_kernels = false;
  for (uint32_t i = 0; i < root_field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_string(&reader, &key));
    if (iree_string_view_equal(key, IREE_SV("amdhsa.kernels"))) {
      if (has_kernels) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU metadata repeats `amdhsa.kernels`");
      }
      has_kernels = true;
      uint32_t kernel_count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_array_count(&reader, &kernel_count));
      out_count->kernel_count = kernel_count;
      for (uint32_t j = 0; j < kernel_count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_count_kernel_args(
            &reader, &out_count->arg_count));
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(&reader, 0));
    }
  }
  if (!has_kernels) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata missing `amdhsa.kernels`");
  }
  if (reader.current != reader.end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata has trailing MessagePack bytes");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_parse_workgroup_size(
    iree_hal_amdgpu_msgpack_reader_t* reader, uint32_t out_value[3]) {
  uint32_t value_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_msgpack_read_array_count(reader, &value_count));
  if (value_count != 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU workgroup size metadata must have three "
                            "elements");
  }
  for (iree_host_size_t i = 0; i < 3; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_msgpack_read_uint32(reader, &out_value[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_parse_arg(
    iree_hal_amdgpu_msgpack_reader_t* reader,
    iree_hal_amdgpu_hsaco_metadata_arg_t* out_arg) {
  memset(out_arg, 0, sizeof(*out_arg));
  iree_hal_amdgpu_hsaco_metadata_arg_fields_t fields = {0};
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_msgpack_read_map_count(reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_string(reader, &key));
    if (iree_string_view_equal(key, IREE_SV(".offset"))) {
      if (fields.has_offset) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_uint32(reader, &out_arg->offset));
      fields.has_offset = true;
    } else if (iree_string_view_equal(key, IREE_SV(".size"))) {
      if (fields.has_size) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_uint32(reader, &out_arg->size));
      fields.has_size = true;
    } else if (iree_string_view_equal(key, IREE_SV(".value_kind"))) {
      if (fields.has_value_kind) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_string(reader, &out_arg->value_kind));
      out_arg->kind =
          iree_hal_amdgpu_hsaco_metadata_classify_arg_kind(out_arg->value_kind);
      fields.has_value_kind = true;
    } else if (iree_string_view_equal(key, IREE_SV(".address_space"))) {
      if (fields.has_address_space) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_string(reader, &out_arg->address_space));
      fields.has_address_space = true;
    } else if (iree_string_view_equal(key, IREE_SV(".access"))) {
      if (fields.has_access) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      iree_string_view_t access = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_string(reader, &access));
      if (!fields.has_actual_access) out_arg->access = access;
      fields.has_access = true;
    } else if (iree_string_view_equal(key, IREE_SV(".actual_access"))) {
      if (fields.has_actual_access) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_string(reader, &out_arg->access));
      fields.has_actual_access = true;
    } else if (iree_string_view_equal(key, IREE_SV(".align")) ||
               iree_string_view_equal(key, IREE_SV(".alignment"))) {
      if (fields.has_alignment) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_uint32(reader, &out_arg->alignment));
      fields.has_alignment = true;
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, 0));
    }
  }
  if (!fields.has_offset || !fields.has_size || !fields.has_value_kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel argument metadata missing required "
                            "offset, size, or value_kind");
  }
  if (out_arg->alignment != 0 &&
      !iree_host_size_is_power_of_two(out_arg->alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel argument alignment must be a power "
                            "of two");
  }
  if (out_arg->alignment != 0 &&
      !iree_host_size_has_alignment(out_arg->offset, out_arg->alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel argument offset is not aligned");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_validate_arg_ranges(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel) {
  for (iree_host_size_t i = 0; i < kernel->arg_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_arg_t* arg = &kernel->args[i];
    iree_host_size_t arg_end = 0;
    if (!iree_host_size_checked_add(arg->offset, arg->size, &arg_end) ||
        arg_end > kernel->kernarg_segment_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU kernel `%.*s` argument %" PRIhsz
                              " exceeds kernarg segment size %u",
                              (int)kernel->symbol_name.size,
                              kernel->symbol_name.data, i,
                              kernel->kernarg_segment_size);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_parse_kernel(
    iree_hal_amdgpu_msgpack_reader_t* reader,
    iree_hal_amdgpu_hsaco_metadata_t* metadata,
    iree_host_size_t* inout_arg_index,
    iree_hal_amdgpu_hsaco_metadata_kernel_t* out_kernel) {
  memset(out_kernel, 0, sizeof(*out_kernel));
  iree_hal_amdgpu_hsaco_metadata_kernel_fields_t fields = {0};
  uint32_t field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_msgpack_read_map_count(reader, &field_count));
  for (uint32_t i = 0; i < field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_string(reader, &key));
    if (iree_string_view_equal(key, IREE_SV(".name"))) {
      if (fields.has_name) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_string(reader, &out_kernel->name));
      fields.has_name = true;
    } else if (iree_string_view_equal(key, IREE_SV(".symbol"))) {
      if (fields.has_symbol_name) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_string(
          reader, &out_kernel->symbol_name));
      fields.has_symbol_name = true;
    } else if (iree_string_view_equal(key, IREE_SV(".kernarg_segment_size"))) {
      if (fields.has_kernarg_segment_size) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_uint32(
          reader, &out_kernel->kernarg_segment_size));
      fields.has_kernarg_segment_size = true;
    } else if (iree_string_view_equal(key, IREE_SV(".kernarg_segment_align"))) {
      if (fields.has_kernarg_segment_alignment) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_uint32(
          reader, &out_kernel->kernarg_segment_alignment));
      fields.has_kernarg_segment_alignment = true;
    } else if (iree_string_view_equal(key,
                                      IREE_SV(".group_segment_fixed_size"))) {
      if (fields.has_group_segment_fixed_size) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_uint32(
          reader, &out_kernel->group_segment_fixed_size));
      fields.has_group_segment_fixed_size = true;
    } else if (iree_string_view_equal(key,
                                      IREE_SV(".private_segment_fixed_size"))) {
      if (fields.has_private_segment_fixed_size) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_uint32(
          reader, &out_kernel->private_segment_fixed_size));
      fields.has_private_segment_fixed_size = true;
    } else if (iree_string_view_equal(key, IREE_SV(".reqd_workgroup_size"))) {
      if (fields.has_required_workgroup_size) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_parse_workgroup_size(
          reader, out_kernel->required_workgroup_size));
      out_kernel->has_required_workgroup_size = true;
      fields.has_required_workgroup_size = true;
    } else if (iree_string_view_equal(key, IREE_SV(".args"))) {
      if (fields.has_args) {
        return iree_hal_amdgpu_hsaco_metadata_duplicate_field_status(key);
      }
      fields.has_args = true;
      uint32_t arg_count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_array_count(reader, &arg_count));
      if (arg_count > metadata->arg_count ||
          *inout_arg_index > metadata->arg_count - arg_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU metadata argument count changed "
                                "between parse passes");
      }
      out_kernel->arg_count = arg_count;
      out_kernel->args = arg_count ? &metadata->args[*inout_arg_index] : NULL;
      for (uint32_t j = 0; j < arg_count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_parse_arg(
            reader, &metadata->args[*inout_arg_index + j]));
      }
      *inout_arg_index += arg_count;
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(reader, 0));
    }
  }

  if (!fields.has_symbol_name || !fields.has_kernarg_segment_size ||
      !fields.has_kernarg_segment_alignment ||
      !fields.has_group_segment_fixed_size ||
      !fields.has_private_segment_fixed_size || !fields.has_args) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel metadata missing required fields");
  }
  if (!iree_host_size_is_power_of_two(out_kernel->kernarg_segment_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel kernarg alignment must be a power "
                            "of two");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_hsaco_metadata_validate_arg_ranges(out_kernel));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_parse_message_pack(
    iree_const_byte_span_t message_pack_data,
    iree_hal_amdgpu_hsaco_metadata_t* metadata) {
  iree_hal_amdgpu_msgpack_reader_t reader = {
      .current = message_pack_data.data,
      .end = message_pack_data.data + message_pack_data.data_length,
  };
  uint32_t root_field_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_msgpack_read_map_count(&reader, &root_field_count));
  bool has_kernels = false;
  iree_host_size_t arg_index = 0;
  for (uint32_t i = 0; i < root_field_count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_read_string(&reader, &key));
    if (iree_string_view_equal(key, IREE_SV("amdhsa.kernels"))) {
      if (has_kernels) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU metadata repeats `amdhsa.kernels`");
      }
      has_kernels = true;
      uint32_t kernel_count = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_msgpack_read_array_count(&reader, &kernel_count));
      if (kernel_count != metadata->kernel_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU metadata kernel count changed between "
                                "parse passes");
      }
      for (uint32_t j = 0; j < kernel_count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_parse_kernel(
            &reader, metadata, &arg_index, &metadata->kernels[j]));
      }
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_msgpack_skip(&reader, 0));
    }
  }
  if (!has_kernels) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata missing `amdhsa.kernels`");
  }
  if (arg_index != metadata->arg_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata argument count changed between "
                            "parse passes");
  }
  if (reader.current != reader.end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata has trailing MessagePack bytes");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_metadata_allocate_storage(
    iree_hal_amdgpu_hsaco_metadata_count_t count,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_hsaco_metadata_t* metadata) {
  metadata->kernel_count = count.kernel_count;
  metadata->arg_count = count.arg_count;
  if (count.kernel_count == 0 && count.arg_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t kernels_size = 0;
  iree_host_size_t args_offset = 0;
  iree_host_size_t args_size = 0;
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(
          count.kernel_count, sizeof(metadata->kernels[0]), &kernels_size) ||
      !iree_host_size_checked_align(
          kernels_size, iree_alignof(iree_hal_amdgpu_hsaco_metadata_arg_t),
          &args_offset) ||
      !iree_host_size_checked_mul(count.arg_count, sizeof(metadata->args[0]),
                                  &args_size) ||
      !iree_host_size_checked_add(args_offset, args_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata storage size overflow");
  }

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&storage));
  memset(storage, 0, total_size);
  metadata->kernels = (iree_hal_amdgpu_hsaco_metadata_kernel_t*)storage;
  metadata->args =
      count.arg_count
          ? (iree_hal_amdgpu_hsaco_metadata_arg_t*)(storage + args_offset)
          : NULL;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
    iree_const_byte_span_t elf_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_hsaco_metadata_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(out_metadata);
  memset(out_metadata, 0, sizeof(*out_metadata));
  out_metadata->host_allocator = host_allocator;
  out_metadata->elf_data = elf_data;

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_metadata_find_note(
      elf_data, &out_metadata->message_pack_data));

  iree_hal_amdgpu_hsaco_metadata_count_t count = {0};
  iree_status_t status = iree_hal_amdgpu_hsaco_metadata_count_message_pack(
      out_metadata->message_pack_data, &count);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_hsaco_metadata_allocate_storage(
        count, host_allocator, out_metadata);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_hsaco_metadata_parse_message_pack(
        out_metadata->message_pack_data, out_metadata);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_hsaco_metadata_deinitialize(out_metadata);
  }
  return status;
}

void iree_hal_amdgpu_hsaco_metadata_deinitialize(
    iree_hal_amdgpu_hsaco_metadata_t* metadata) {
  if (!metadata) return;
  if (metadata->kernels) {
    iree_allocator_free(metadata->host_allocator, metadata->kernels);
  }
  memset(metadata, 0, sizeof(*metadata));
}

iree_status_t iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
    const iree_hal_amdgpu_hsaco_metadata_t* metadata,
    iree_string_view_t symbol_name,
    const iree_hal_amdgpu_hsaco_metadata_kernel_t** out_kernel) {
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(out_kernel);
  *out_kernel = NULL;
  for (iree_host_size_t i = 0; i < metadata->kernel_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel =
        &metadata->kernels[i];
    if (!iree_string_view_equal(kernel->symbol_name, symbol_name)) continue;
    if (*out_kernel) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU metadata has duplicate kernel symbol "
                              "`%.*s`",
                              (int)symbol_name.size, symbol_name.data);
    }
    *out_kernel = kernel;
  }
  if (!*out_kernel) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU metadata kernel symbol `%.*s` not found",
                            (int)symbol_name.size, symbol_name.data);
  }
  return iree_ok_status();
}
