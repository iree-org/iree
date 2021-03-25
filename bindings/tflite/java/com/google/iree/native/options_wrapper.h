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

#ifndef BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_OPTIONS_WRAPPER_H_
#define BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_OPTIONS_WRAPPER_H_

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"

namespace iree {
namespace tflite {

class OptionsWrapper {
 public:
  OptionsWrapper();
  ~OptionsWrapper();

  TfLiteInterpreterOptions* options() const { return options_; }

  void SetNumThreads(int num_threads);

 private:
  TfLiteInterpreterOptions* options_ = nullptr;
};

}  // namespace tflite
}  // namespace iree

#endif  // BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_OPTIONS_WRAPPER_H_
