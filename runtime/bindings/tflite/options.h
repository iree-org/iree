// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_TFLITE_OPTIONS_H_
#define IREE_BINDINGS_TFLITE_OPTIONS_H_

#include "iree/base/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

struct TfLiteInterpreterOptions {
  iree_allocator_t allocator;
  int32_t num_threads;
  void (*reporter)(void* user_data, const char* format, va_list args);
  void* reporter_user_data;

  // TODO(#3977): an existing iree_hal_device_t to use for the HAL.
};

void _TfLiteInterpreterOptionsSetDefaults(TfLiteInterpreterOptions* options);

#endif  // IREE_BINDINGS_TFLITE_OPTIONS_H_
