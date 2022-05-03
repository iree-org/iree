// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_TFLITE_SHIM_H_
#define IREE_BINDINGS_TFLITE_SHIM_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

TfLiteStatus _TfLiteStatusFromIREEStatus(iree_status_t status);

#endif  // IREE_BINDINGS_TFLITE_SHIM_H_
