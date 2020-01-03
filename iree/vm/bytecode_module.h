// Copyright 2019 Google LLC
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

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a VM module from an in-memory ModuleDef FlatBuffer.
// If a |flatbuffer_allocator| is provided then it will be used to free the
// |flatbuffer_data| when the module is destroyed and otherwise the ownership of
// the flatbuffer_data remains with the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_bytecode_module_create(
    iree_const_byte_span_t flatbuffer_data,
    iree_allocator_t flatbuffer_allocator, iree_allocator_t allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_MODULE_H_
