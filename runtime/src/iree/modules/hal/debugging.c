// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/debugging.h"

//===----------------------------------------------------------------------===//
// Debug Sink
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_null(void) {
  iree_hal_module_debug_sink_t sink = {0};
  return sink;
}

#if defined(IREE_FILE_IO_ENABLE)

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
static iree_status_t iree_hal_module_buffer_view_trace_stdio(
    void* user_data, iree_string_view_t key, iree_host_size_t buffer_view_count,
    iree_hal_buffer_view_t** buffer_views, iree_allocator_t host_allocator) {
  FILE* file = (FILE*)user_data;

  fprintf(file, "=== %.*s ===\n", (int)key.size, key.data);
  for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
    iree_hal_buffer_view_t* buffer_view = buffer_views[i];

    // NOTE: this export is for debugging only and a no-op in min-size builds.
    // We heap-alloc here because at the point this export is used performance
    // is not a concern.

    // Query total length (excluding NUL terminator).
    iree_host_size_t result_length = 0;
    iree_status_t status = iree_hal_buffer_view_format(
        buffer_view, IREE_HOST_SIZE_MAX, 0, NULL, &result_length);
    if (!iree_status_is_out_of_range(status)) {
      return status;
    }
    ++result_length;  // include NUL

    // Allocate scratch heap memory to contain the result and format into it.
    char* result_str = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, result_length,
                                               (void**)&result_str));
    status =
        iree_hal_buffer_view_format(buffer_view, IREE_HOST_SIZE_MAX,
                                    result_length, result_str, &result_length);
    if (iree_status_is_ok(status)) {
      fprintf(file, "%.*s\n", (int)result_length, result_str);
    }
    iree_allocator_free(host_allocator, result_str);
    IREE_RETURN_IF_ERROR(status);
  }
  fprintf(file, "\n");
  return iree_ok_status();
}
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_stdio(FILE* file) {
  iree_hal_module_debug_sink_t sink = {0};

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
  sink.buffer_view_trace.fn = iree_hal_module_buffer_view_trace_stdio;
  sink.buffer_view_trace.user_data = file;
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

  return sink;
}

#endif  // IREE_FILE_IO_ENABLE
