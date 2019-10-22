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

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_VM_API_H_
#define IREE_VM_API_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/rt/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree::vm::BytecodeModule
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a VM module from an in-memory ModuleDef FlatBuffer.
// The provided |buffer_free_fn| will be called when the module is destroyed
// and only if this creation function succeeds. If ownership remains with the
// caller then pass nullptr for |buffer_free_fn|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_bytecode_module_create_from_buffer(
    iree_const_byte_span_t buffer_data,
    void (*buffer_free_fn)(void* self, iree_byte_span_t buffer_data),
    void* buffer_free_self, iree_allocator_t allocator,
    iree_rt_module_t** out_module);

// Creates a VM module from a mapped ModuleDef FlatBuffer.
// The provided |file_mapping| will be retained for the life of the module and
// the contents will be accessed by reference.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_bytecode_module_create_from_file_mapping(
    iree_file_mapping_t* file_mapping, iree_allocator_t allocator,
    iree_rt_module_t** out_module);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_API_H_
