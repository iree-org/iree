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

#ifndef BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_INTERPRETER_WRAPPER_H_
#define BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_INTERPRETER_WRAPPER_H_

#include <memory>

#include "bindings/tflite/java/com/google/iree/native/model_wrapper.h"
#include "bindings/tflite/java/com/google/iree/native/options_wrapper.h"
#include "bindings/tflite/java/com/google/iree/native/tensor_wrapper.h"
#include "iree/base/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"

namespace iree {
namespace tflite {

class InterpreterWrapper {
 public:
  iree_status_t Create(const ModelWrapper& model,
                       const OptionsWrapper& options);
  ~InterpreterWrapper();

  TfLiteInterpreter* interpreter() const { return interpreter_; }
  int get_input_tensor_count() const {
    return TfLiteInterpreterGetInputTensorCount(interpreter_);
  }
  int get_output_tensor_count() const {
    return TfLiteInterpreterGetOutputTensorCount(interpreter_);
  }

  iree_status_t AllocateTensors();
  iree_status_t ResizeInputTensor(int32_t input_index, const int* input_dims,
                                  int32_t input_dims_size);
  std::unique_ptr<TensorWrapper> GetInputTensor(int input_index);
  std::unique_ptr<TensorWrapper> GetOutputTensor(int output_index);
  iree_status_t Invoke();

 private:
  TfLiteInterpreter* interpreter_ = nullptr;
};

}  // namespace tflite
}  // namespace iree

#endif  // BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_INTERPRETER_WRAPPER_H_
