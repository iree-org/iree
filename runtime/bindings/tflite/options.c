// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime/bindings/tflite/options.h"

#include "iree/base/tracing.h"
#include "runtime/bindings/tflite/shim.h"

void _TfLiteInterpreterOptionsSetDefaults(TfLiteInterpreterOptions* options) {
  options->num_threads = -1;
}

TFL_CAPI_EXPORT extern TfLiteInterpreterOptions*
TfLiteInterpreterOptionsCreate() {
  iree_allocator_t allocator = iree_allocator_system();
  IREE_TRACE_ZONE_BEGIN(z0);

  TfLiteInterpreterOptions* options = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*options), (void**)&options);
  if (!iree_status_is_ok(iree_status_consume_code(status))) {
    IREE_TRACE_MESSAGE(ERROR, "failed options allocation");
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }
  memset(options, 0, sizeof(*options));
  options->allocator = allocator;
  _TfLiteInterpreterOptionsSetDefaults(options);

  IREE_TRACE_ZONE_END(z0);
  return options;
}

TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(options->allocator, options);
  IREE_TRACE_ZONE_END(z0);
}

TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options, int32_t num_threads) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, num_threads);
  options->num_threads = num_threads;
  IREE_TRACE_ZONE_END(z0);
}

TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsAddDelegate(
    TfLiteInterpreterOptions* options, TfLiteDelegate* delegate) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Silently ignored as if it never tried to take an ops for itself.
  IREE_TRACE_MESSAGE(WARNING,
                     "TfLiteInterpreterOptionsAddDelegate: delegates are "
                     "unsupported and ignored in the IREE tflite shim");

  IREE_TRACE_ZONE_END(z0);
}

TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  options->reporter = reporter;
  options->reporter_user_data = user_data;
  IREE_TRACE_ZONE_END(z0);
}

TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetUseNNAPI(
    TfLiteInterpreterOptions* options, bool enable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Silently ignored as if it wasn't present.
  if (enable) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "enabled", strlen("enabled"));
    IREE_TRACE_MESSAGE(WARNING,
                       "TfLiteInterpreterOptionsSetUseNNAPI: the NNAPI is "
                       "unsupported and ignored in the IREE tflite shim");
  } else {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "enabled", strlen("disabled"));
  }

  IREE_TRACE_ZONE_END(z0);
}
