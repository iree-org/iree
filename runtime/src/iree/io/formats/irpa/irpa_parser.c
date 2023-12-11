// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_parser.h"

#include "iree/schemas/parameter_archive.h"

static iree_status_t iree_io_verify_irpa_v0_file_range(
    iree_const_byte_span_t file_contents, iree_io_physical_offset_t base_offset,
    iree_io_parameter_archive_range_t range) {
  if (range.length == 0) return iree_ok_status();
  if (base_offset + range.offset + range.length > file_contents.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file segment out of range (%" PRIu64 " to %" PRIu64
                            " for %" PRIu64 ", file_size=%" PRIhsz ")",
                            base_offset + range.offset,
                            base_offset + range.offset + range.length - 1,
                            range.length, file_contents.data_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_io_resolve_irpa_v0_string(
    iree_const_byte_span_t file_contents, iree_io_physical_offset_t base_offset,
    const iree_io_parameter_archive_header_v0_t* header,
    iree_io_parameter_archive_metadata_ref_t range,
    iree_string_view_t* out_view) {
  *out_view = iree_string_view_empty();
  if (range.length == 0) return iree_ok_status();
  if (range.offset + range.length > header->metadata_segment.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "metadata segment reference out of range (%" PRIu64
                            " to %" PRIu64 " for %" PRIu64
                            ", segment_size=%" PRIu64 ")",
                            range.offset, range.offset + range.length - 1,
                            range.length, header->metadata_segment.length);
  }
  iree_io_physical_offset_t view_offset =
      base_offset + header->metadata_segment.offset + range.offset;
  *out_view =
      iree_make_string_view(file_contents.data + view_offset, range.length);
  return iree_ok_status();
}

static iree_status_t iree_io_resolve_irpa_v0_metadata(
    iree_const_byte_span_t file_contents, iree_io_physical_offset_t base_offset,
    const iree_io_parameter_archive_header_v0_t* header,
    iree_io_parameter_archive_metadata_ref_t range,
    iree_const_byte_span_t* out_span) {
  *out_span = iree_const_byte_span_empty();
  if (range.length == 0) return iree_ok_status();
  if (range.offset + range.length > header->metadata_segment.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "metadata segment reference out of range (%" PRIu64
                            " to %" PRIu64 " for %" PRIu64
                            ", segment_size=%" PRIu64 ")",
                            range.offset, range.offset + range.length - 1,
                            range.length, header->metadata_segment.length);
  }
  iree_io_physical_offset_t span_offset =
      base_offset + header->metadata_segment.offset + range.offset;
  *out_span =
      iree_make_const_byte_span(file_contents.data + span_offset, range.length);
  return iree_ok_status();
}

static iree_status_t iree_io_resolve_irpa_v0_storage(
    iree_const_byte_span_t file_contents, iree_io_physical_offset_t base_offset,
    const iree_io_parameter_archive_header_v0_t* header,
    iree_io_parameter_archive_storage_ref_t range,
    iree_io_physical_offset_t* out_offset) {
  *out_offset = 0;
  if (range.length == 0) return iree_ok_status();
  if (range.offset + range.length > header->storage_segment.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage segment reference out of range (%" PRIu64
                            " to %" PRIu64 " for %" PRIu64
                            ", segment_size=%" PRIu64 ")",
                            range.offset, range.offset + range.length - 1,
                            range.length, header->storage_segment.length);
  }
  iree_io_physical_offset_t storage_offset =
      base_offset + header->storage_segment.offset + range.offset;
  *out_offset = storage_offset;
  return iree_ok_status();
}

static iree_status_t iree_io_parse_irpa_v0_splat_entry(
    const iree_io_parameter_archive_header_v0_t* header,
    const iree_io_parameter_archive_splat_entry_t* splat_entry,
    iree_string_view_t name, iree_const_byte_span_t metadata,
    iree_io_parameter_index_t* index) {
  if (splat_entry->header.entry_size < sizeof(*splat_entry)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "splat entry length underflow");
  }
  if (splat_entry->pattern_length > sizeof(splat_entry->pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat pattern length %u out of bounds %" PRIhsz,
                            splat_entry->pattern_length,
                            sizeof(splat_entry->pattern));
  }
  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = splat_entry->length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage =
          {
              .splat =
                  {
                      .pattern = {0},  // set below
                      .pattern_length = splat_entry->pattern_length,
                  },
          },
  };
  memcpy(entry.storage.splat.pattern, splat_entry->pattern,
         entry.storage.splat.pattern_length);
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t iree_io_parse_irpa_v0_data_entry(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_physical_offset_t base_offset,
    const iree_io_parameter_archive_header_v0_t* header,
    const iree_io_parameter_archive_data_entry_t* data_entry,
    iree_string_view_t name, iree_const_byte_span_t metadata,
    iree_io_parameter_index_t* index) {
  if (data_entry->header.entry_size < sizeof(*data_entry)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "data entry length underflow");
  }
  iree_io_physical_offset_t storage_offset = 0;
  IREE_RETURN_IF_ERROR(
      iree_io_resolve_irpa_v0_storage(file_contents, base_offset, header,
                                      data_entry->storage, &storage_offset));
  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = data_entry->storage.length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
      .storage =
          {
              .file =
                  {
                      .handle = file_handle,
                      .offset = storage_offset,
                  },
          },
  };
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t iree_io_parse_irpa_v0_index_from_memory(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_physical_offset_t base_offset,
    const iree_io_parameter_archive_header_prefix_t* header_prefix,
    iree_io_parameter_index_t* index) {
  // Get the full header struct.
  if (header_prefix->version_minor > 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "IRPA version %u.%u not supported (major supported "
        "but minor is newer than the runtime trying to parse it)",
        header_prefix->version_major, header_prefix->version_minor);
  }
  if (header_prefix->header_size !=
      sizeof(iree_io_parameter_archive_header_v0_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IRPA v0 header expected to be exactly %" PRIhsz
                            " bytes but was reported as %" PRIu64,
                            sizeof(iree_io_parameter_archive_header_v0_t),
                            header_prefix->header_size);
  }
  const iree_io_parameter_archive_header_v0_t* header =
      (const iree_io_parameter_archive_header_v0_t*)file_contents.data;

  // Verify the base data ranges; this lets all subsequent checks be against the
  // header instead of needing to know about the view into the file.
  IREE_RETURN_IF_ERROR(iree_io_verify_irpa_v0_file_range(
                           file_contents, base_offset, header->entry_segment),
                       "verifying entry table");
  IREE_RETURN_IF_ERROR(
      iree_io_verify_irpa_v0_file_range(file_contents, base_offset,
                                        header->metadata_segment),
      "verifying metadata segment");
  IREE_RETURN_IF_ERROR(iree_io_verify_irpa_v0_file_range(
                           file_contents, base_offset, header->storage_segment),
                       "verifying storage segment");

  // Walk the entry table, which has variable-length entries.
  iree_io_physical_offset_t entry_offset =
      base_offset + header->entry_segment.offset;
  iree_io_physical_size_t entry_size_remaining = header->entry_segment.length;
  for (iree_io_physical_size_t i = 0; i < header->entry_count; ++i) {
    // Ensure there's enough space in the table for the base entry header.
    if (entry_size_remaining <
        sizeof(iree_io_parameter_archive_entry_header_t)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "entry table truncated; insufficient bytes for base entry header");
    }

    // Ensure there's enough space for the declared entry size (if any larger).
    const iree_io_parameter_archive_entry_header_t* entry_header =
        (const iree_io_parameter_archive_entry_header_t*)(file_contents.data +
                                                          entry_offset);
    if (entry_header->entry_size < sizeof(*entry_header) ||
        entry_size_remaining < entry_header->entry_size) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "entry table truncated; insufficient bytes for sized header");
    }
    // TODO(benvanik): make this explicit with iree_io_stream_seek_to_alignment.
    iree_io_physical_offset_t aligned_entry_size = iree_align_uint64(
        entry_header->entry_size, IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT);
    entry_offset += aligned_entry_size;
    entry_size_remaining -= aligned_entry_size;

    // Resolve entry metadata from the archive metadata segment.
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_io_resolve_irpa_v0_string(file_contents, base_offset, header,
                                       entry_header->name, &name),
        "resolving entry name");
    iree_const_byte_span_t metadata = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(
        iree_io_resolve_irpa_v0_metadata(file_contents, base_offset, header,
                                         entry_header->metadata, &metadata),
        "resolving entry metadata");

    // Handle each entry type.
    switch (entry_header->type) {
      case IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SKIP:
        break;
      case IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT: {
        IREE_RETURN_IF_ERROR(iree_io_parse_irpa_v0_splat_entry(
            header,
            (const iree_io_parameter_archive_splat_entry_t*)entry_header, name,
            metadata, index));
        break;
      }
      case IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA: {
        IREE_RETURN_IF_ERROR(iree_io_parse_irpa_v0_data_entry(
            file_handle, file_contents, base_offset, header,
            (const iree_io_parameter_archive_data_entry_t*)entry_header, name,
            metadata, index));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "parser does not support entry type %d",
                                (int)entry_header->type);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_io_parse_irpa_index_from_memory(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_physical_offset_t base_offset, iree_io_parameter_index_t* index) {
  // Check the basic header information is something we can process.
  if (file_contents.data_length <
      base_offset + sizeof(iree_io_parameter_archive_header_prefix_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "not enough bytes for a valid IRPA header; file "
                            "may be empty or truncated");
  }
  const iree_io_parameter_archive_header_prefix_t* header_prefix =
      (const iree_io_parameter_archive_header_prefix_t*)(file_contents.data +
                                                         base_offset);
  if (header_prefix->magic != IREE_IO_PARAMETER_ARCHIVE_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IRPA file magic missing or invalid %08X; expected %08X",
        header_prefix->magic, IREE_IO_PARAMETER_ARCHIVE_MAGIC);
  }
  if (header_prefix->header_size > file_contents.data_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "file buffer underrun parsing header of reported size %" PRIu64
        " (only %" PRIhsz " bytes available)",
        header_prefix->header_size, file_contents.data_length);
  }
  if (header_prefix->next_header_offset != 0 &&
      file_contents.data_length <
          base_offset + header_prefix->next_header_offset +
              sizeof(iree_io_parameter_archive_header_prefix_t)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "file buffer underrun verifying linked header at offset %" PRIu64
        " (only %" PRIhsz " bytes available)",
        base_offset + header_prefix->next_header_offset,
        file_contents.data_length);
  }

  // Route major versions to their parsers, allowing us to change everything but
  // the prefix without breaking compatibility.
  switch (header_prefix->version_major) {
    case 0: {
      IREE_RETURN_IF_ERROR(iree_io_parse_irpa_v0_index_from_memory(
          file_handle, file_contents, base_offset, header_prefix, index));
      break;
    }
    default: {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "IRPA major version %u.%u not supported by this runtime",
          header_prefix->version_major, header_prefix->version_minor);
    }
  }

  // If there's a linked header then tail-call process it.
  if (header_prefix->next_header_offset == 0) return iree_ok_status();
  return iree_io_parse_irpa_index_from_memory(
      file_handle, file_contents,
      base_offset + header_prefix->next_header_offset, index);
}

IREE_API_EXPORT iree_status_t iree_io_parse_irpa_index(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Today we only support memory files.
  // TODO(benvanik): support iree_io_stream_t wrapping for parsing the index.
  if (iree_io_file_handle_type(file_handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-memory irpa files not yet supported");
  }
  iree_byte_span_t host_allocation =
      iree_io_file_handle_primitive(file_handle).value.host_allocation;

  iree_status_t status = iree_io_parse_irpa_index_from_memory(
      file_handle,
      iree_make_const_byte_span(host_allocation.data,
                                host_allocation.data_length),
      /*base_offset=*/0, index);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
