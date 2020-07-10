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

#include "bindings/java/com/google/iree/native/module_wrapper.h"

#include "iree/base/api_util.h"

namespace iree {
namespace java {

Status ModuleWrapper::Create(const uint8_t* flatbuffer_data,
                             iree_host_size_t length) {
  return FromApiStatus(
      iree_vm_bytecode_module_create(
          iree_const_byte_span_t{flatbuffer_data, length}, IREE_ALLOCATOR_NULL,
          IREE_ALLOCATOR_SYSTEM, &module_),
      IREE_LOC);
}

ModuleWrapper::~ModuleWrapper() { iree_vm_module_release(module_); }

}  // namespace java
}  // namespace iree
