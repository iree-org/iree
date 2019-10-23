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

#include "vm/api.h"

#include "base/api.h"
#include "base/api_util.h"
#include "base/flatbuffer_util.h"
#include "base/tracing.h"
#include "vm/sequencer_module.h"

namespace iree {
namespace vm {

//===----------------------------------------------------------------------===//
// iree::vm::BytecodeModule
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_bytecode_module_create_from_buffer(
    iree_const_byte_span_t buffer_data,
    void (*buffer_free_fn)(void* self, iree_byte_span_t buffer_data),
    void* buffer_free_self, iree_allocator_t allocator,
    iree_rt_module_t** out_module) {
  IREE_TRACE_SCOPE0("iree_vm_bytecode_module_create_from_buffer");

  if (!out_module) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_module = nullptr;

  if (!buffer_data.data || !buffer_data.data_length) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto module_file,
      FlatBufferFile<ModuleDef>::FromBuffer(
          ModuleDefIdentifier(), {buffer_data.data, buffer_data.data_length},
          [buffer_free_fn, buffer_free_self, buffer_data]() {
            if (buffer_free_fn != nullptr) {
              buffer_free_fn(buffer_free_self,
                             {const_cast<uint8_t*>(buffer_data.data),
                              buffer_data.data_length});
            }
          }));

  IREE_API_ASSIGN_OR_RETURN(auto module,
                            SequencerModule::FromFile(std::move(module_file)));

  *out_module = reinterpret_cast<iree_rt_module_t*>(module.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_bytecode_module_create_from_file_mapping(
    iree_file_mapping_t* file_mapping, iree_allocator_t allocator,
    iree_rt_module_t** out_module) {
  IREE_TRACE_SCOPE0("iree_vm_bytecode_module_create_from_file_mapping");

  if (!out_module) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_module = nullptr;

  if (!file_mapping) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto buffer_data = iree_file_mapping_data(file_mapping);
  IREE_API_ASSIGN_OR_RETURN(
      auto module_file,
      FlatBufferFile<ModuleDef>::FromBuffer(
          ModuleDefIdentifier(), {buffer_data.data, buffer_data.data_length},
          [file_mapping]() { iree_file_mapping_release(file_mapping); }));
  iree_file_mapping_retain(file_mapping);

  IREE_API_ASSIGN_OR_RETURN(auto module,
                            SequencerModule::FromFile(std::move(module_file)));

  *out_module = reinterpret_cast<iree_rt_module_t*>(module.release());

  return IREE_STATUS_OK;
}

}  // namespace vm
}  // namespace iree
