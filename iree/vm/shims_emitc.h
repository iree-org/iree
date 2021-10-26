// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SHIMS_EMITC_H_
#define IREE_VM_SHIMS_EMITC_H_

#include "iree/base/attributes.h"
#include "iree/vm/module.h"
#include "iree/vm/shims.h"
#include "iree/vm/stack.h"

// see calling convention in module.h
// Variadic arguments are not supported

// 0v_v
typedef iree_status_t (*call_0v_v_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state);

static iree_status_t call_0v_v_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0v_v_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  return target_fn(stack, module, module_state);
}

// 0v_i
typedef iree_status_t (*call_0v_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state, int32_t* res0);

static iree_status_t call_0v_i_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0v_i_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  iree_vm_abi_i_t* rets = iree_vm_abi_i_checked_deref(call->results);

  if (IREE_UNLIKELY(!rets)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }

  iree_vm_abi_i_reset(rets);
  return target_fn(stack, module, module_state, &rets->i0);
}

// 0i_i
typedef iree_status_t (*call_0i_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state, int32_t arg0,
                                     int32_t* res0);

static iree_status_t call_0i_i_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0i_i_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  const iree_vm_abi_i_t* args = iree_vm_abi_i_checked_deref(call->arguments);
  iree_vm_abi_i_t* rets = iree_vm_abi_i_checked_deref(call->results);

  if (IREE_UNLIKELY(!args || !rets)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }

  iree_vm_abi_i_reset(rets);
  return target_fn(stack, module, module_state, args->i0, &rets->i0);
}

static iree_status_t call_0i_i_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      int32_t arg0, int32_t* out_ret0) {
  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arg0, sizeof(arg0));
  call.results = iree_make_byte_span(out_ret0, sizeof(*out_ret0));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  return import->module->begin_call(import->module, stack, &call, &result);
}

// 0ii_i
typedef iree_status_t (*call_0ii_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                      void* module_state, int32_t arg0,
                                      int32_t arg1, int32_t* res0);

static iree_status_t call_0ii_i_shim(iree_vm_stack_t* stack,
                                     const iree_vm_function_call_t* call,
                                     call_0ii_i_t target_fn, void* module,
                                     void* module_state,
                                     iree_vm_execution_result_t* out_result) {
  const iree_vm_abi_ii_t* args = iree_vm_abi_ii_checked_deref(call->arguments);
  iree_vm_abi_i_t* rets = iree_vm_abi_i_checked_deref(call->results);

  if (IREE_UNLIKELY(!args || !rets)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }

  iree_vm_abi_i_reset(rets);
  return target_fn(stack, module, module_state, args->i0, args->i1, &rets->i0);
}

#endif  // IREE_VM_SHIMS_EMITC_H_
