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

#ifndef IREE_BINDINGS_TFLITE_SHIM_H_
#define IREE_BINDINGS_TFLITE_SHIM_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"
#include "bindings/tflite/include/tensorflow/lite/c/c_api_experimental.h"

TfLiteStatus _TfLiteStatusFromIREEStatus(iree_status_t status);

#endif  // IREE_BINDINGS_TFLITE_SHIM_H_
