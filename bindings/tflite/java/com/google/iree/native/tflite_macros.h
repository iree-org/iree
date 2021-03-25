// Copyright 2021 Google LLC
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

#ifndef BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_TFLITE_MACROS_H_
#define BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_TFLITE_MACROS_H_

#include "iree/base/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"

#define TFLITE_TO_IREE_STATUS(tf_status) \
  ((tf_status) == kTfLiteOk)             \
      ? iree_ok_status()                 \
      : iree_make_status(IREE_STATUS_ABORTED, "TFLite Error")

#define TFLITE_RETURN_IF_ERROR(tf_status) \
  IREE_RETURN_IF_ERROR(TFLITE_TO_IREE_STATUS(tf_status))

#endif  // BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_TFLITE_MACROS_H_
