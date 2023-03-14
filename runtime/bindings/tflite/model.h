// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_TFLITE_MODEL_H_
#define IREE_BINDINGS_TFLITE_MODEL_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/vm/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

typedef struct _TfLiteModelExports {
  iree_vm_function_t _reset_variables;
  iree_vm_function_t _query_input_shape;
  iree_vm_function_t _resize_input_shape;
  iree_vm_function_t _query_output_shape;
  iree_vm_function_t _main;
} _TfLiteModelExports;

struct TfLiteModel {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;
  void* owned_model_data;

  // HACK: no public API that allows us to share this without spooky action
  // at a distance. Today it's ok for these to be unique as we don't check that
  // they are consistent and all instances have the same types registered.
  iree_vm_instance_t* instance;
  iree_vm_module_t* module;
  _TfLiteModelExports exports;
  int32_t input_count;
  int32_t output_count;
};

void _TfLiteModelRetain(TfLiteModel* model);
void _TfLiteModelRelease(TfLiteModel* model);

#endif  // IREE_BINDINGS_TFLITE_MODEL_H_
