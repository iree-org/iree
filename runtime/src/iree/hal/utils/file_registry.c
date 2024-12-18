// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/file_registry.h"

#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/memory_file.h"

IREE_API_EXPORT iree_status_t iree_hal_file_from_handle(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  switch (iree_io_file_handle_type(handle)) {
    case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION:
      status =
          iree_hal_memory_file_wrap(device_allocator, queue_affinity, access,
                                    handle, host_allocator, out_file);
      break;
    case IREE_IO_FILE_HANDLE_TYPE_FD:
      status = iree_hal_fd_file_from_handle(access, handle, host_allocator,
                                            out_file);
      break;
    default:
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "no common implementation supported for file handles of type %d",
          (int)iree_io_file_handle_type(handle));
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
