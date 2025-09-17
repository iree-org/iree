// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/archive.h"

#include "iree/vm/bytecode/utils/isa.h"

// ZIP local file header (comes immediately before each file in the archive).
// In order to find the starting offset of the FlatBuffer in a polyglot archive
// we need to parse this given the variable-length nature of it (we want to
// be robust to file name and alignment changes).
//
// NOTE: all fields are little-endian.
// NOTE: we don't care about the actual module size here; since we support
//       streaming archives trying to recover it would require much more
//       involved processing (we'd need to reference the central directory).
//       If we wanted to support users repacking ZIPs we'd probably want to
//       rewrite everything as we store offsets in the FlatBuffer that are
//       difficult to update after the archive has been produced.
#define ZIP_LOCAL_FILE_HEADER_SIGNATURE 0x04034B50u
#if defined(IREE_COMPILER_MSVC)
#pragma pack(push, 1)
#endif  // IREE_COMPILER_MSVC
typedef struct {
  uint32_t signature;  // ZIP_LOCAL_FILE_HEADER_SIGNATURE
  uint16_t version;
  uint16_t general_purpose_flag;
  uint16_t compression_method;
  uint16_t last_modified_time;
  uint16_t last_modified_date;
  uint32_t crc32;              // 0 for us
  uint32_t compressed_size;    // 0 for us
  uint32_t uncompressed_size;  // 0 for us
  uint16_t file_name_length;
  uint16_t extra_field_length;
  // file name (variable size)
  // extra field (variable size)
} IREE_ATTRIBUTE_PACKED zip_local_file_header_t;
#if defined(IREE_COMPILER_MSVC)
#pragma pack(pop)
#endif  // IREE_COMPILER_MSVC
static_assert(sizeof(zip_local_file_header_t) == 30, "bad packing");
#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "little endian required for zip header parsing"
#endif  // IREE_ENDIANNESS_LITTLE

// Strips any ZIP local file header from |contents| and stores the remaining
// range in |out_stripped|.
static iree_status_t iree_vm_bytecode_module_strip_zip_header(
    iree_const_byte_span_t contents, iree_const_byte_span_t* out_stripped) {
  // Ensure there's at least some bytes we can check for the header.
  // Since we're only looking to strip zip stuff here we can check on that.
  if (!contents.data ||
      contents.data_length < sizeof(zip_local_file_header_t)) {
    memmove(out_stripped, &contents, sizeof(contents));
    return iree_ok_status();
  }

  // Check to see if there's a zip local header signature.
  // For a compliant zip file this is expected to start at offset 0.
  const zip_local_file_header_t* header =
      (const zip_local_file_header_t*)contents.data;
  if (header->signature != ZIP_LOCAL_FILE_HEADER_SIGNATURE) {
    // No signature found, probably not a ZIP.
    memmove(out_stripped, &contents, sizeof(contents));
    return iree_ok_status();
  }

  // Compute the starting offset of the file.
  // Note that we still don't know (or care) if it's the file we want; actual
  // FlatBuffer verification happens later on.
  const uint32_t offset =
      sizeof(*header) + header->file_name_length + header->extra_field_length;
  if (offset > contents.data_length) {
    // Is a ZIP but doesn't have enough data; error out with something more
    // useful than the FlatBuffer verification failing later on given that here
    // we know this isn't a FlatBuffer.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "archive self-reports as a zip but does not have "
                            "enough data to contain a module");
  }

  *out_stripped = iree_make_const_byte_span(contents.data + offset,
                                            contents.data_length - offset);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_archive_parse_header(
    iree_const_byte_span_t archive_contents,
    iree_const_byte_span_t* out_flatbuffer_contents,
    iree_host_size_t* out_rodata_offset) {
  // Slice off any polyglot zip header we have prior to the base of the module.
  iree_const_byte_span_t module_contents = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_strip_zip_header(
      archive_contents, &module_contents));

  // Verify there's enough data to safely check the FlatBuffer header.
  if (!module_contents.data || module_contents.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "FlatBuffer data is not present or less than 16 bytes (%" PRIhsz
        " total)",
        module_contents.data_length);
  }

  // Read the size prefix from the head of the module contents; this should be
  // a 4 byte value indicating the total size of the FlatBuffer data.
  size_t length_prefix = 0;
  flatbuffers_read_size_prefix((void*)module_contents.data, &length_prefix);

  // Verify the length prefix is within bounds (always <= the remaining module
  // bytes).
  const size_t length_remaining =
      module_contents.data_length - sizeof(flatbuffers_uoffset_t);
  if (length_prefix > length_remaining) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FlatBuffer length prefix out of bounds (prefix is "
                            "%zu but only %zu available)",
                            length_prefix, length_remaining);
  }

  // Form the range of bytes containing just the FlatBuffer data.
  iree_const_byte_span_t flatbuffer_contents = iree_make_const_byte_span(
      module_contents.data + sizeof(flatbuffers_uoffset_t), length_prefix);

  if (out_flatbuffer_contents) {
    *out_flatbuffer_contents = flatbuffer_contents;
  }
  if (out_rodata_offset) {
    // rodata begins immediately following the FlatBuffer in memory.
    const iree_host_size_t rodata_offset = iree_host_align(
        (iree_host_size_t)(flatbuffer_contents.data - archive_contents.data) +
            length_prefix,
        IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT);
    *out_rodata_offset = rodata_offset;
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_bytecode_archive_infer_size(iree_const_byte_span_t archive_contents,
                                    iree_host_size_t* out_inferred_size) {
  IREE_ASSERT_ARGUMENT(out_inferred_size);
  *out_inferred_size = 0;

  // Handle the case where size is unknown (unsafe inference).
  const bool unsafe_infer_size = (archive_contents.data_length == 0);
  if (unsafe_infer_size) {
    archive_contents.data_length = IREE_HOST_SIZE_MAX;
  }

  // Parse header to find the relative offset we need for calculating the
  // extents.
  iree_const_byte_span_t flatbuffer_contents = iree_const_byte_span_empty();
  iree_host_size_t rodata_offset = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_archive_parse_header(
      archive_contents, &flatbuffer_contents, &rodata_offset));

  // Parse the BytecodeModuleDef to find rodata segments.
  iree_vm_BytecodeModuleDef_table_t module_def =
      iree_vm_BytecodeModuleDef_as_root(flatbuffer_contents.data);
  if (!module_def) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse BytecodeModuleDef");
  }

  // Find the maximum extent of all rodata segments.
  iree_vm_RodataSegmentDef_vec_t rodata_segments =
      iree_vm_BytecodeModuleDef_rodata_segments(module_def);
  iree_host_size_t rodata_length = 0;
  for (size_t i = 0; i < iree_vm_RodataSegmentDef_vec_len(rodata_segments);
       ++i) {
    iree_vm_RodataSegmentDef_table_t segment =
        iree_vm_RodataSegmentDef_vec_at(rodata_segments, i);
    if (segment &&
        iree_vm_RodataSegmentDef_external_data_length_is_present(segment)) {
      const uint64_t offset =
          iree_vm_RodataSegmentDef_external_data_offset(segment);
      const uint64_t length =
          iree_vm_RodataSegmentDef_external_data_length(segment);
      const uint64_t segment_end = offset + length;
      if (segment_end > rodata_length) {
        rodata_length = segment_end;  // bump extent
      }
    }
  }

  // Total size is rodata_offset + rodata_length.
  // There may be ZIP headers behind this but we don't currently care.
  *out_inferred_size = rodata_offset + rodata_length;
  return iree_ok_status();
}
