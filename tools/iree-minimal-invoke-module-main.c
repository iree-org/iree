// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal C program linking in the IREE runtime. Does not even take enough
// command-line parameters to be able to successfully run most modules. Only
// useful for code size/symbols monitoring purposes.

#include <stdio.h>

// Low-level IREE VM APIs.
// The higher-level iree/runtime/api.h can be used for more complete ML-like
// programs using the hardware abstraction layer (HAL). This simple sample just
// uses base VM types.
#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

// NOTE: CHECKs are dangerous but this is a sample; a real application would
// want to handle errors gracefully. We know in this constrained case that
// these won't fail unless something is catastrophically wrong (out of memory,
// solar flares, etc).
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage:\n"
            "  %s </path/to/say_hello.vmfb> <entry.point>\n",
            argv[0]);
    fprintf(
        stderr,
        "This program performs a dummy invoke without actual parameters, so "
        "the function call will fail in general. It is only meant to be used "
        "to inspect IREE runtime code size and internal call graph.\n");
    return -1;
  }

  // Internally IREE does not (in general) use malloc and instead uses the
  // provided allocator to allocate and free memory. Applications can integrate
  // their own allocator as-needed.
  iree_allocator_t allocator = iree_allocator_system();

  // Create the root isolated VM instance that we can create contexts within.
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(allocator, &instance));
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance));

  // Load the module from stdin or a file on disk.
  // Applications can ship and load modules however they want (such as mapping
  // them into memory instead of allocating like this). Modules can also be
  // embedded in the binary but in those cases it makes more sense to use emitc
  // to avoid the bytecode entirely and have a fully static build (see
  // samples/emitc_modules/ for some examples).
  const char* module_path = argv[1];
  iree_file_contents_t* module_contents = NULL;
  IREE_CHECK_OK(
      iree_file_read_contents(module_path, allocator, &module_contents));

  // Load the bytecode module from the vmfb.
  // This module can be reused across multiple contexts.
  // Note that we let the module retain the file contents for as long as needed.
  iree_vm_module_t* bytecode_module = NULL;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance, module_contents->const_buffer,
      iree_file_contents_deallocator(module_contents), allocator,
      &bytecode_module));

  // Create the context for this invocation reusing the loaded modules.
  // Contexts hold isolated state and can be reused for multiple calls.
  // Note that the module order matters: the input user module is dependent on
  // the custom module.
  iree_vm_module_t* user_modules[] = {bytecode_module};
  iree_tooling_module_list_t modules;
  iree_tooling_module_list_initialize(&modules);
  IREE_CHECK_OK(iree_tooling_resolve_modules(
      instance, IREE_ARRAYSIZE(user_modules), user_modules,
      /*default_device_uri=*/iree_string_view_empty(), allocator, &modules,
      /*out_device=*/NULL, /*out_device_allocator=*/NULL));

  iree_vm_context_t* context = NULL;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, modules.count, modules.values,
      allocator, &context));

  // Lookup the function by fully-qualified name (module.func).
  iree_vm_function_t function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(argv[2]), &function));

  fprintf(stdout, "INVOKE BEGIN %s\n", argv[2]);
  fflush(stdout);

  // Synchronously invoke the requested function.
  // We don't pass in/out anything in these simple examples so the I/O lists
  // are not needed.
  IREE_CHECK_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                               /*policy=*/NULL, /*inputs=*/NULL,
                               /*outputs=*/NULL, allocator));

  fprintf(stdout, "INVOKE END\n");
  fflush(stdout);

  iree_tooling_module_list_reset(&modules);
  iree_vm_context_release(context);
  iree_vm_module_release(bytecode_module);
  iree_vm_instance_release(instance);
  return 0;
}
