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

#include "bindings/tflite/java/com/google/iree/native/interpreter_wrapper.h"

#include "absl/memory/memory.h"
#include "bindings/tflite/java/com/google/iree/native/tflite_macros.h"

namespace iree {
namespace tflite {

iree_status_t InterpreterWrapper::Create(const ModelWrapper& model,
                                         const OptionsWrapper& options) {
  interpreter_ = TfLiteInterpreterCreate(model.model(), options.options());
  if (interpreter_ == nullptr) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "Failed to create interpreter wrapper.");
  }
  return iree_ok_status();
}

InterpreterWrapper::~InterpreterWrapper() {
  TfLiteInterpreterDelete(interpreter_);
  interpreter_ = nullptr;
}

iree_status_t InterpreterWrapper::AllocateTensors() {
  TFLITE_RETURN_IF_ERROR(TfLiteInterpreterAllocateTensors(interpreter_));
  return iree_ok_status();
}

iree_status_t InterpreterWrapper::ResizeInputTensor(int32_t input_index,
                                                    const int* input_dims,
                                                    int32_t input_dims_size) {
  return TFLITE_TO_IREE_STATUS(TfLiteInterpreterResizeInputTensor(
      interpreter_, input_index, input_dims, input_dims_size));
}

std::unique_ptr<TensorWrapper> InterpreterWrapper::GetInputTensor(
    int input_index) {
  return absl::make_unique<TensorWrapper>(
      TfLiteInterpreterGetInputTensor(interpreter_, input_index));
}

std::unique_ptr<TensorWrapper> InterpreterWrapper::GetOutputTensor(
    int output_index) {
  return absl::make_unique<TensorWrapper>(
      TfLiteInterpreterGetOutputTensor(interpreter_, output_index));
}

iree_status_t InterpreterWrapper::Invoke() {
  return TFLITE_TO_IREE_STATUS(TfLiteInterpreterInvoke(interpreter_));
}

}  // namespace tflite
}  // namespace iree
