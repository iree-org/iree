// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/executable_header.h"

iree_status_t iree_hal_read_executable_flatbuffer_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    const char file_identifier[4],
    iree_const_byte_span_t* out_flatbuffer_data) {
  iree_flatbuffer_file_header_t header = {0};

  // Check minimum size for flatbuffer (size prefix + minimal header).
  if (!unsafe_infer_size && executable_data.data_length < sizeof(header)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable flatbuffer data is not present or less than header size");
  }

  // Check the magic first; if we supported other file types we'd use this
  // to switch out to other loaders (maybe). Unless the magic matches we can't
  // assume there's 64 bytes to read.
  if (memcmp(executable_data.data, file_identifier, sizeof(header.magic)) !=
      0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable file identifier does not match; expected `%.*s`", 4,
        file_identifier);
  }

  // Read the header from the executable data.
  // This ensures alignment so we can directly access the fields (though the
  // data itself _should_ be aligned).
  memcpy(&header, executable_data.data, sizeof(header));

  // Zero-length flatbuffers are invalid.
  if (header.content_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable flatbuffer reported as zero-length");
  }

  // Verify version. Today we have only a single "latest" version so this is
  // just a guard for older builds with newer data that may differ in the
  // future.
  if (header.version != 0) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "executable version %u not compatible with this build", header.version);
  }

  // Verify the size is within bounds.
  if (!unsafe_infer_size) {
    const iree_host_size_t remaining_length =
        executable_data.data_length - sizeof(header);
    if (header.content_size > remaining_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "executable flatbuffer size prefix out of bounds (size is %" PRIu64
          " but "
          "only %" PRIhsz " bytes available)",
          header.content_size, remaining_length);
    }
  }

  // Adjust the flatbuffer data to exclude the size prefix.
  *out_flatbuffer_data = iree_make_const_byte_span(
      executable_data.data + sizeof(header), header.content_size);

  return iree_ok_status();
}
