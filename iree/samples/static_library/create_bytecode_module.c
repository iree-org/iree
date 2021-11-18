// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <stdio.h>

#include "iree/samples/static_library/simple_mul_c.h"
#include "iree/vm/bytecode_module.h"

// A function to create the bytecode module.
iree_status_t create_module(iree_vm_module_t** module) {
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_static_library_simple_mul_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);

  return iree_vm_bytecode_module_create(module_data, iree_allocator_null(),
                                        iree_allocator_system(), module);
}

void print_success() { printf("static_library_run_bytecode passed\n"); }
