// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime/bindings/tflite/shim.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

TFL_CAPI_EXPORT extern const char* TfLiteVersion(void) { return "👻"; }

TfLiteStatus _TfLiteStatusFromIREEStatus(iree_status_t status) {
  switch (iree_status_consume_code(status)) {
    case IREE_STATUS_OK:
      return kTfLiteOk;
    default:
      return kTfLiteError;
  }
}
