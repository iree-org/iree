// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_BINDINGS_TFLITE_INTERPRETER_H_
#define IREE_BINDINGS_TFLITE_INTERPRETER_H_

#include "bindings/tflite/model.h"
#include "bindings/tflite/options.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

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
