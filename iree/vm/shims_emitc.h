// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SHIMS_EMITC_H_
#define IREE_VM_SHIMS_EMITC_H_

#include "iree/base/attributes.h"
#include "iree/vm/module.h"
#include "iree/vm/stack.h"

typedef iree_status_t (*iree_vm_native_function_target_emitc)(
    iree_vm_stack_t* IREE_RESTRICT stack,
    iree_vm_function_call_t* IREE_RESTRICT call, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state,
    iree_vm_execution_result_t* IREE_RESTRICT);

static iree_status_t iree_emitc_shim(
    iree_vm_stack_t* IREE_RESTRICT stack,
    /*const*/ iree_vm_function_call_t* IREE_RESTRICT call,
    iree_vm_native_function_target_emitc target_fn, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state,
    iree_vm_execution_result_t* IREE_RESTRICT out_result) {
  return target_fn(stack, call, module, module_state, out_result);
}

#endif  // IREE_VM_SHIMS_EMITC_H_
