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

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/bytecode_module_size_benchmark_module.h"

extern "C" int main(int argc, char** argv) {
  iree_vm_instance_t* instance = nullptr;
  iree_vm_instance_create(iree_allocator_system(), &instance);

  const auto* module_file_toc =
      iree::vm::bytecode_module_size_benchmark_module_create();
  iree_vm_module_t* module = nullptr;
  iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &module);

  iree_vm_context_t* context = nullptr;
  iree_vm_context_create_with_modules(instance, &module, /*module_count=*/1,
                                      iree_allocator_system(), &context);

  iree_vm_function_t function;
  iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("empty_func"), &function);

  iree_vm_invoke(context, function, /*policy=*/nullptr, /*inputs=*/nullptr,
                 /*outputs=*/nullptr, iree_allocator_system());

  iree_vm_module_release(module);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  return 0;
}
