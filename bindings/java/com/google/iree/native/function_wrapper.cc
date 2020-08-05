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

#include "bindings/java/com/google/iree/native/function_wrapper.h"

namespace iree {
namespace java {

iree_vm_function_t* FunctionWrapper::function() const {
  return function_.get();
}

iree_string_view_t FunctionWrapper::name() const {
  return iree_vm_function_name(function_.get());
}

iree_vm_function_signature_t FunctionWrapper::signature() const {
  return iree_vm_function_signature(function_.get());
}

}  // namespace java
}  // namespace iree
