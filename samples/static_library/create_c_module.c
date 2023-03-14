// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <stdio.h>

#include "samples/static_library/simple_mul_emitc.h"

// A function to create the C module.
iree_status_t create_module(iree_vm_instance_t* instance,
                            iree_vm_module_t** out_module) {
  return module_create(instance, iree_vm_instance_allocator(instance),
                       out_module);
}

void print_success() { printf("static_library_run_c passed\n"); }
