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
// We use the format {version}_{arguments}_{results}
// Variadic arguments are not supported

// 0.
typedef iree_status_t (*call_0___t)(iree_vm_stack_t* stack, void* module_ptr,
                                    void* module_state);

static iree_status_t call_0___shim(iree_vm_stack_t* stack,
                                   const iree_vm_function_call_t* call,
                                   call_0___t target_fn, void* module,
                                   void* module_state,
                                   iree_vm_execution_result_t* out_result) {
  return target_fn(stack, module, module_state);
}

#endif  // IREE_VM_TEST_EMITC_SHIMS_H_
