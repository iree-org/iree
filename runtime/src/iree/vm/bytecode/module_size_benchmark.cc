// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_size_benchmark_module_c.h"

extern "C" int main(int argc, char** argv) {
  iree_vm_instance_t* instance = nullptr;
  iree_vm_instance_create(iree_allocator_system(), &instance);

  const auto* module_file_toc =
      iree_vm_bytecode_module_size_benchmark_module_create();
  iree_vm_module_t* module = nullptr;
  iree_vm_bytecode_module_create(
      instance,
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &module);

  iree_vm_context_t* context = nullptr;
  iree_vm_context_create_with_modules(instance, IREE_VM_CONTEXT_FLAG_NONE,
                                      /*module_count=*/1, &module,
                                      iree_allocator_system(), &context);

  iree_vm_function_t function;
  iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("empty_func"), &function);

  iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                 /*policy=*/nullptr, /*inputs=*/nullptr,
                 /*outputs=*/nullptr, iree_allocator_system());

  iree_vm_module_release(module);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  return 0;
}
