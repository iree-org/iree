// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_builder.h"

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_initialize(
    iree_allocator_t host_allocator,
    iree_io_parameter_archive_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->file_alignment =
      IREE_IO_PARAMETER_ARCHIVE_DEFAULT_FILE_ALIGNMENT;
  return iree_io_parameter_index_create(host_allocator, &out_builder->index);
}

IREE_API_EXPORT void iree_io_parameter_archive_builder_deinitialize(
    iree_io_parameter_archive_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_io_parameter_index_release(builder->index);
  memset(builder, 0, sizeof(*builder));
}

static iree_io_physical_size_t
iree_io_parameter_archive_builder_storage_alignment(
    const iree_io_parameter_archive_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  return builder->storage_alignment ? builder->storage_alignment : 1;
}

static iree_io_physical_offset_t
iree_io_parameter_archive_builder_storage_offset(
    const iree_io_parameter_archive_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  return iree_align_uint64(
      iree_align_uint64(sizeof(iree_io_parameter_archive_header_v0_t),
                        IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT) +
          builder->entry_segment_size + builder->metadata_segment_size,
      iree_io_parameter_archive_builder_storage_alignment(builder));
}

IREE_API_EXPORT iree_io_physical_size_t
iree_io_parameter_archive_builder_total_size(
    const iree_io_parameter_archive_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  return iree_align_uint64(
      iree_io_parameter_archive_builder_storage_offset(builder) +
          builder->storage_segment_size,
      builder->file_alignment);
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_write(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_file_handle_t* file_handle, iree_io_physical_offset_t file_offset,
    iree_io_stream_t* stream, iree_io_parameter_index_t* target_index) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(target_index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate relative segment offsets in the file based on the expected header
  // size. This allows us to emit header-relative file offsets in the header
  // itself but absolute offsets for consumers of the target_index.
  const iree_io_parameter_archive_range_t entry_segment = {
      .offset = iree_align_uint64(sizeof(iree_io_parameter_archive_header_v0_t),
                                  IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT),
      .length = builder->entry_segment_size,
  };
  const iree_io_parameter_archive_range_t metadata_segment = {
      .offset = entry_segment.offset + entry_segment.length,
      .length = builder->metadata_segment_size,
  };
  const iree_io_parameter_archive_range_t storage_segment = {
      .offset = iree_align_uint64(
          metadata_segment.offset + metadata_segment.length,
          iree_io_parameter_archive_builder_storage_alignment(builder)),
      .length = builder->storage_segment_size,
  };

  // Write the archive header referencing the other segments in the file.
  iree_io_parameter_archive_header_v0_t header = {
      .prefix =
          {
              .magic = IREE_IO_PARAMETER_ARCHIVE_MAGIC,
              .version_major = 0,
              .version_minor = 0,
              .header_size = sizeof(header),
              .next_header_offset = 0,
              .flags = 0,
          },
      .entry_count = iree_io_parameter_index_count(builder->index),
      .entry_segment = entry_segment,
      .metadata_segment = metadata_segment,
      .storage_segment = storage_segment,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_write(stream, sizeof(header), &header));

  // Write entry table following the header.
  // This references ranges in the metadata and storage segment but to preserve
  // forward-only writes we populate those after writing the table.
  iree_io_physical_offset_t metadata_offset = 0;
  for (iree_host_size_t i = 0;
       i < iree_io_parameter_index_count(builder->index); ++i) {
    // Align each entry to the base alignment.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_seek_to_alignment(
                stream, IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT));

    // Query the source entry template.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_index_get(builder->index, i, &source_entry));

    // Reserve space in the metadata segment.
    const iree_io_parameter_archive_metadata_ref_t name_ref = {
        .offset = metadata_offset,
        .length = source_entry->key.size,
    };
    metadata_offset += name_ref.length;
    const iree_io_parameter_archive_metadata_ref_t metadata_ref = {
        .offset = source_entry->metadata.data_length ? metadata_offset : 0,
        .length = source_entry->metadata.data_length,
    };
    metadata_offset += metadata_ref.length;

    // Produce the target archive entry based on the template.
    iree_io_parameter_index_entry_t target_entry = {
        .key = source_entry->key,
        .metadata = source_entry->metadata,
        .length = source_entry->length,
        .type = source_entry->type,
        .storage = source_entry->storage,
    };
    switch (source_entry->type) {
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
        iree_io_parameter_archive_splat_entry_t splat_entry = {
            .header =
                {
                    .entry_size = sizeof(splat_entry),
                    .type = IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT,
                    .flags = 0,
                    .name = name_ref,
                    .metadata = metadata_ref,
                    .minimum_alignment = 0,
                },
            .length = target_entry.length,
            .pattern_length = target_entry.storage.splat.pattern_length,
        };
        memcpy(splat_entry.pattern, target_entry.storage.splat.pattern,
               sizeof(splat_entry.pattern));
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_io_stream_write(stream, sizeof(splat_entry), &splat_entry));
        break;
      }
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
        iree_io_parameter_archive_data_entry_t data_entry = {
            .header =
                {
                    .entry_size = sizeof(data_entry),
                    .type = IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA,
                    .flags = 0,
                    .name = name_ref,
                    .metadata = metadata_ref,
                    .minimum_alignment =
                        IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
                },
            .storage =
                {
                    .offset = target_entry.storage.file.offset,
                    .length = target_entry.length,
                },
        };
        target_entry.storage.file.handle = file_handle;
        target_entry.storage.file.offset += storage_segment.offset;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_io_stream_write(stream, sizeof(data_entry), &data_entry));
        break;
      }
      default: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                 "unhandled entry type %d",
                                 (int)source_entry->type));
      }
    }

    // Add the entry to the target_index referencing the location in the file
    // reserved for the entry storage.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_index_add(target_index, &target_entry));
  }

  // Write out the metadata table.
  for (iree_host_size_t i = 0;
       i < iree_io_parameter_index_count(builder->index); ++i) {
    // Query the source entry template.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_index_get(builder->index, i, &source_entry));

    // Write header metadata.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_write(stream, source_entry->key.size,
                                 source_entry->key.data));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_write(stream, source_entry->metadata.data_length,
                                 source_entry->metadata.data));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_add_splat_entry(
    iree_io_parameter_archive_builder_t* builder, iree_string_view_t name,
    iree_const_byte_span_t metadata, const void* pattern,
    uint8_t pattern_length, iree_io_physical_size_t data_length) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(pattern);
  if (pattern_length == 0 ||
      pattern_length > IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH ||
      !iree_is_power_of_two_uint64(pattern_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "splat pattern length %u invalid; must be 1, 2, 4, 8, or 16 bytes",
        pattern_length);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);
  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = data_length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage =
          {
              .splat =
                  {
                      .pattern = {0},  // set below
                      .pattern_length = pattern_length,
                  },
          },
  };
  memcpy(entry.storage.splat.pattern, pattern, pattern_length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_add(builder->index, &entry));
  builder->entry_segment_size =
      iree_align_uint64(builder->entry_segment_size,
                        IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT) +
      sizeof(iree_io_parameter_archive_splat_entry_t);
  builder->metadata_segment_size += name.size + metadata.data_length;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_add_data_entry(
    iree_io_parameter_archive_builder_t* builder, iree_string_view_t name,
    iree_const_byte_span_t metadata, iree_io_physical_size_t minimum_alignment,
    iree_io_physical_size_t data_length) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);
  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = data_length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
      .storage =
          {
              .file =
                  {
                      .handle = NULL,  // set on commit
                      .offset = iree_align_uint64(builder->storage_segment_size,
                                                  minimum_alignment),
                  },
          },
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_add(builder->index, &entry));
  builder->entry_segment_size =
      iree_align_uint64(builder->entry_segment_size,
                        IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT) +
      sizeof(iree_io_parameter_archive_data_entry_t);
  builder->metadata_segment_size += name.size + metadata.data_length;
  builder->storage_segment_size = entry.storage.file.offset + entry.length;
  if (!builder->storage_alignment) {
    // First entry sets the base alignment.
    builder->storage_alignment = minimum_alignment;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_build_parameter_archive(
    iree_io_parameter_index_t* source_index,
    iree_io_parameter_index_t* target_index,
    iree_io_parameter_archive_file_open_callback_t target_file_open,
    iree_io_physical_offset_t target_file_offset,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(source_index);
  IREE_ASSERT_ARGUMENT(target_index);
  IREE_ASSERT_ARGUMENT(target_file_open.fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_archive_builder_t builder;
  iree_io_parameter_archive_builder_initialize(host_allocator, &builder);

  // Declare a parameter for each entry in the index.
  // This lets us calculate the size we require to store the entry metadata and
  // its contents (if any). No data is accessed yet.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_io_parameter_index_count(source_index);
       ++i) {
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    status = iree_io_parameter_index_get(source_index, i, &source_entry);
    if (!iree_status_is_ok(status)) break;
    switch (source_entry->type) {
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
        status = iree_io_parameter_archive_builder_add_splat_entry(
            &builder, source_entry->key, source_entry->metadata,
            source_entry->storage.splat.pattern,
            source_entry->storage.splat.pattern_length, source_entry->length);
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
        status = iree_io_parameter_archive_builder_add_data_entry(
            &builder, source_entry->key, source_entry->metadata,
            IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
            source_entry->length);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unhandled index entry storage type %d",
                                  (int)source_entry->type);
        break;
    }
    if (!iree_status_is_ok(status)) break;
  }

  // Open a file of sufficient size (now that we know it) for writing.
  iree_io_physical_offset_t archive_offset = iree_align_uint64(
      target_file_offset, IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT);
  iree_io_physical_size_t archive_length =
      iree_io_parameter_archive_builder_total_size(&builder);
  iree_io_file_handle_t* target_file_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = target_file_open.fn(target_file_open.user_data, archive_offset,
                                 archive_length, &target_file_handle);
  }

  // Wrap the target file in a stream.
  iree_io_stream_t* target_stream = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, target_file_handle,
                            target_file_offset, host_allocator, &target_stream);
  }

  // Commit the archive header to the file and produce an index referencing it.
  // This will allow us to know where to copy file contents.
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_archive_builder_write(
        &builder, target_file_handle, target_file_offset, target_stream,
        target_index);
  }

  // Copy over parameter entry file contents (if any).
  // This is a slow operation and something we could optimize with lower-level
  // platform primitives.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < iree_io_parameter_index_count(source_index); ++i) {
      const iree_io_parameter_index_entry_t* source_entry = NULL;
      status = iree_io_parameter_index_get(source_index, i, &source_entry);
      if (!iree_status_is_ok(status)) break;
      const iree_io_parameter_index_entry_t* target_entry = NULL;
      status = iree_io_parameter_index_lookup(target_index, source_entry->key,
                                              &target_entry);
      if (!iree_status_is_ok(status)) break;
      switch (source_entry->type) {
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
          // No work to do.
          break;
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
          status = iree_io_stream_seek(
              target_stream, IREE_IO_STREAM_SEEK_SET,
              target_file_offset + target_entry->storage.file.offset);
          if (!iree_status_is_ok(status)) break;
          status = iree_io_stream_write_file(
              target_stream, source_entry->storage.file.handle,
              source_entry->storage.file.offset, target_entry->length,
              host_allocator);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "unhandled index entry storage type %d",
                                    (int)source_entry->type);
          break;
      }
      if (!iree_status_is_ok(status)) break;
    }
  }

  iree_io_stream_release(target_stream);

  // Flush file contents before returning to the caller (in case they open the
  // file via a different handle).
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_flush(target_file_handle);
  }

  iree_io_file_handle_release(target_file_handle);
  iree_io_parameter_archive_builder_deinitialize(&builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
