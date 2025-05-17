// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_DEBUGGING_H_
#define IREE_MODULES_HAL_DEBUGGING_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Debug Sink
//===----------------------------------------------------------------------===//

typedef void(IREE_API_PTR* iree_hal_module_debug_sink_release_callback_fn_t)(
    void* user_data);

typedef struct iree_hal_module_debug_sink_release_callback_t {
  iree_hal_module_debug_sink_release_callback_fn_t fn;
  void* user_data;
} iree_hal_module_debug_sink_release_callback_t;

typedef iree_status_t(
    IREE_API_PTR* iree_hal_module_buffer_view_trace_callback_fn_t)(
    void* user_data, iree_string_view_t key, iree_host_size_t buffer_view_count,
    iree_hal_buffer_view_t** buffer_views, iree_allocator_t host_allocator);

typedef struct iree_hal_module_buffer_view_trace_callback_t {
  iree_hal_module_buffer_view_trace_callback_fn_t fn;
  void* user_data;
} iree_hal_module_buffer_view_trace_callback_t;

// Interface for a HAL module debug event sink.
// Any referenced user data must remain live until all users of the sink have
// released it via the release callback.
typedef struct iree_hal_module_debug_sink_t {
  // Called once per HAL module that has been provided the debug sink when it
  // will no longer be used. Optional if there's no lifetime management of user
  // data required.
  iree_hal_module_debug_sink_release_callback_t release;
  // Called on each hal.buffer_view.trace.
  iree_hal_module_buffer_view_trace_callback_t buffer_view_trace;
} iree_hal_module_debug_sink_t;

// Returns a default debug sink that outputs nothing.
IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_null(void);

#if IREE_FILE_IO_ENABLE

// Returns a default debug sink that routes to an stdio stream in textual form.
IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_stdio(FILE* file);

#else

#define iree_hal_module_debug_sink_stdio(file) iree_hal_module_debug_sink_null()

#endif  // IREE_FILE_IO_ENABLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_DEBUGGING_H_
