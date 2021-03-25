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

#ifndef BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_TENSOR_WRAPPER_H_
#define BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_TENSOR_WRAPPER_H_

#include "iree/base/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"

namespace iree {
namespace tflite {

class TensorWrapper {
 public:
  TensorWrapper(const TfLiteTensor* tensor);
  TensorWrapper(TfLiteTensor* mutable_tensor_);
  ~TensorWrapper();

  const TfLiteTensor* tensor() const {
    return is_mutable_tensor() ? mutable_tensor_ : tensor_;
  }
  bool is_mutable_tensor() const { return mutable_tensor_ != nullptr; }
  TfLiteType tensor_type() const { return TfLiteTensorType(tensor()); }
  int num_dims() const { return TfLiteTensorNumDims(tensor()); }
  int dim(int dim_index) const { return TfLiteTensorDim(tensor(), dim_index); }
  size_t byte_size() const { return TfLiteTensorByteSize(tensor()); }
  void* tensor_data() const { return TfLiteTensorData(tensor()); }
  const char* tensor_name() const { return TfLiteTensorName(tensor()); }

  TfLiteQuantizationParams quantization_params() const {
    return TfLiteTensorQuantizationParams(tensor());
  }

  iree_status_t CopyFromBuffer(const void* input_data, size_t input_data_size);
  iree_status_t CopyToBuffer(void* output_data, size_t output_data_size);

  void Assign(const TfLiteTensor* tensor);
  void AssignMutable(TfLiteTensor* tensor);

 private:
  const TfLiteTensor* tensor_ = nullptr;
  TfLiteTensor* mutable_tensor_ = nullptr;
};

}  // namespace tflite
}  // namespace iree

#endif  // BINDINGS_TFLITE_JAVA_COM_GOOGLE_IREE_NATIVE_TENSOR_WRAPPER_H_
