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

#ifndef IREE_VM_TEST_EMITC_ARITHMETIC_OPS_H_
#define IREE_VM_TEST_EMITC_ARITHMETIC_OPS_H_

#include "iree/vm/test/emitc/arithmetic_ops.module"
#include "iree/vm/test/emitc/shims.h"

struct arithmetic_ops_s;
struct arithmetic_ops_state_s;
typedef struct arithmetic_ops_s arithmetic_ops_t;
typedef struct arithmetic_ops_state_s arithmetic_ops_state_t;

static iree_status_t arithmetic_ops_test_add_i32(
    iree_vm_stack_t* stack, arithmetic_ops_t* module,
    arithmetic_ops_state_t* module_state) {
  return arithmetic_ops_test_add_i32_impl();
}

static const iree_vm_native_export_descriptor_t arithmetic_ops_exports_[] = {
    {iree_make_cstring_view("test_add_i32"), iree_make_cstring_view("0."), 0,
     NULL},
};

static const iree_vm_native_function_ptr_t arithmetic_ops_funcs_[] = {
    {(iree_vm_native_function_shim_t)call_0___shim,
     (iree_vm_native_function_target_t)arithmetic_ops_test_add_i32},
};

static_assert(IREE_ARRAYSIZE(arithmetic_ops_funcs_) ==
                  IREE_ARRAYSIZE(arithmetic_ops_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t arithmetic_ops_descriptor_ = {
    iree_make_cstring_view("arithmetic_ops"),
    IREE_ARRAYSIZE(arithmetic_ops_imports_),
    arithmetic_ops_imports_,
    IREE_ARRAYSIZE(arithmetic_ops_exports_),
    arithmetic_ops_exports_,
    IREE_ARRAYSIZE(arithmetic_ops_funcs_),
    arithmetic_ops_funcs_,
    0,
    NULL,
};

static iree_status_t arithmetic_ops_create(iree_allocator_t allocator,
                                           iree_vm_module_t** out_module) {
  // NOTE: this module has neither shared or per-context module state.
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  return iree_vm_native_module_create(&interface, &arithmetic_ops_descriptor_,
                                      allocator, out_module);
}

#endif  // IREE_VM_TEST_EMITC_ARITHMETIC_OPS_H_
