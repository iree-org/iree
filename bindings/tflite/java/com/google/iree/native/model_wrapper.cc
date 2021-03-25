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

#include "bindings/tflite/java/com/google/iree/native/model_wrapper.h"

namespace iree {
namespace tflite {

iree_status_t ModelWrapper::Create(const void* model_data, size_t model_size) {
  model_ = TfLiteModelCreate(model_data, model_size);
  if (model_ == nullptr) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "Failed to create model wrapper.");
  }
  return iree_ok_status();
}

ModelWrapper::~ModelWrapper() {
  TfLiteModelDelete(model_);
  model_ = nullptr;
}

}  // namespace tflite
}  // namespace iree
