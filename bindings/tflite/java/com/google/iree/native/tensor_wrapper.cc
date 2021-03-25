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

#include "bindings/tflite/java/com/google/iree/native/tensor_wrapper.h"

#include "bindings/tflite/java/com/google/iree/native/tflite_macros.h"
#include "iree/base/logging.h"

namespace iree {
namespace tflite {

TensorWrapper::TensorWrapper(const TfLiteTensor* tensor) : tensor_(tensor) {}

TensorWrapper::TensorWrapper(TfLiteTensor* mutable_tensor)
    : mutable_tensor_(mutable_tensor) {}

TensorWrapper::~TensorWrapper() { tensor_ = nullptr; }

iree_status_t TensorWrapper::CopyFromBuffer(const void* input_data,
                                            size_t input_data_size) {
  IREE_CHECK(is_mutable_tensor()) << "Can only copy into non-const tensors.";
  TFLITE_RETURN_IF_ERROR(
      TfLiteTensorCopyFromBuffer(mutable_tensor_, input_data, input_data_size));
  return iree_ok_status();
}

iree_status_t TensorWrapper::CopyToBuffer(void* output_data,
                                          size_t output_data_size) {
  TFLITE_RETURN_IF_ERROR(
      TfLiteTensorCopyToBuffer(tensor(), output_data, output_data_size));
  return iree_ok_status();
}

void TensorWrapper::Assign(const TfLiteTensor* tensor) {
  IREE_CHECK_EQ(mutable_tensor_, nullptr) << "This tensor is already mutable.";
  tensor_ = tensor;
}

void TensorWrapper::AssignMutable(TfLiteTensor* tensor) {
  IREE_CHECK_EQ(tensor_, nullptr) << "This tensor is already const.";

  mutable_tensor_ = tensor;
}

}  // namespace tflite
}  // namespace iree
