// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/internal.h"

iree_status_t iree_profile_file_open(iree_string_view_t path,
                                     iree_allocator_t host_allocator,
                                     iree_profile_file_t* out_profile_file) {
  memset(out_profile_file, 0, sizeof(*out_profile_file));

  iree_status_t status =
      iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ, host_allocator,
                                &out_profile_file->contents);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(
        out_profile_file->contents->const_buffer, &out_profile_file->header,
        &out_profile_file->first_record_offset);
  }

  if (!iree_status_is_ok(status)) {
    iree_profile_file_close(out_profile_file);
  }
  return status;
}

void iree_profile_file_close(iree_profile_file_t* profile_file) {
  iree_io_file_contents_free(profile_file->contents);
  memset(profile_file, 0, sizeof(*profile_file));
}

iree_status_t iree_profile_file_for_each_record(
    const iree_profile_file_t* profile_file,
    iree_profile_file_record_callback_t callback, void* user_data) {
  iree_host_size_t record_offset = profile_file->first_record_offset;
  iree_host_size_t record_index = 0;
  while (record_offset < profile_file->contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_profile_file_parse_record(
        profile_file->contents->const_buffer, record_offset, &record,
        &next_record_offset));
    IREE_RETURN_IF_ERROR(callback(user_data, &record, record_index));
    record_offset = next_record_offset;
    ++record_index;
  }
  return iree_ok_status();
}
