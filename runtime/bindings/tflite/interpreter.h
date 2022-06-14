// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_TFLITE_INTERPRETER_H_
#define IREE_BINDINGS_TFLITE_INTERPRETER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "runtime/bindings/tflite/model.h"
#include "runtime/bindings/tflite/options.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

struct TfLiteInterpreter {
  iree_allocator_t allocator;

  TfLiteModel* model;  // retained
  TfLiteInterpreterOptions options;

  iree_vm_instance_t* instance;
  iree_hal_driver_t* driver;
  iree_hal_device_t* device;

  union {
    // NOTE: order matters; later modules in the list resolve symbols using the
    // earlier modules (like system/custom modules).
    struct {
      iree_vm_module_t* hal_module;
      iree_vm_module_t* user_module;
    };
    iree_vm_module_t* all_modules[2];
  };
  iree_vm_context_t* context;

  iree_vm_list_t* input_list;
  iree_vm_list_t* output_list;
  TfLiteTensor* input_tensors;
  TfLiteTensor* output_tensors;
};

#endif  // IREE_BINDINGS_TFLITE_INTERPRETER_H_
