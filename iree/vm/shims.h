// Copyright 2021 Google LLC
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

#ifndef IREE_VM_TEST_EMITC_SHIMS_H_
#define IREE_VM_TEST_EMITC_SHIMS_H_

#include "iree/vm/module.h"
#include "iree/vm/stack.h"

// see Calling convetion in module.h
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

// 0i_i
typedef iree_status_t (*call_0i_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state, int32_t arg0,
                                     int32_t* res0);

static iree_status_t call_0i_i_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0i_i_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  typedef struct {
    int32_t arg0;
  } args_t;
  typedef struct {
    int32_t ret0;
  } results_t;

  const args_t* args = (const args_t*)call->arguments.data;
  results_t* results = (results_t*)call->results.data;

  return target_fn(stack, module, module_state, args->arg0, &results->ret0);
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
  typedef struct {
    int32_t arg0;
    int32_t arg1;
  } args_t;
  typedef struct {
    int32_t ret0;
  } results_t;

  const args_t* args = (const args_t*)call->arguments.data;
  results_t* results = (results_t*)call->results.data;
  return target_fn(stack, module, module_state, args->arg0, args->arg1,
                   &results->ret0);
}

#endif  // IREE_VM_TEST_EMITC_SHIMS_H_
