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

#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/native_module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

// This would be generated together with the functions in the header
#include "iree/samples/emitc_modules/add_module.module"

struct add_module_s;
struct add_module_state_s;
typedef struct add_module_s add_module_t;
typedef struct add_module_state_s add_module_state_t;

static iree_status_t add_module_add_1(add_module_t* module,
                                      add_module_state_t* state,
                                      iree_vm_stack_t* stack,
                                      const iree_vm_function_call_t* call,
                                      iree_vm_execution_result_t* out_result) {
  // TODO(benvanik): iree_vm_stack native frame enter/leave.
  // By not enter/leaving a frame here we won't be able to set breakpoints or
  // tracing on the function. Fine for now.
  iree_vm_stack_frame_t* caller_frame = iree_vm_stack_current_frame(stack);
  const iree_vm_register_list_t* arg_list = call->argument_registers;
  const iree_vm_register_list_t* ret_list = call->result_registers;
  auto& regs = caller_frame->registers;

  // Load the input argument.
  // This should really be generated code (like module_abi_cc.h).
  int32_t arg0 = regs.i32[arg_list->registers[0] & regs.i32_mask];
  int32_t arg1 = regs.i32[arg_list->registers[1] & regs.i32_mask];

  int32_t out;

  add_module_add_1_impl(arg0, arg1, &out);

  // Store the result.
  regs.i32[ret_list->registers[0] & regs.i32_mask] = out;

  return iree_ok_status();
}

static iree_status_t add_module_call_function(
    add_module_t* module, add_module_state_t* state, iree_vm_stack_t* stack,
    const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result) {
  // TODO(benvanik): iree_vm_stack native frame enter/leave.
  // By not enter/leaving a frame here we won't be able to set breakpoints or
  // tracing on the function. Fine for now.
  iree_vm_stack_frame_t* caller_frame = iree_vm_stack_current_frame(stack);
  const iree_vm_register_list_t* arg_list = call->argument_registers;
  const iree_vm_register_list_t* ret_list = call->result_registers;
  auto& regs = caller_frame->registers;

  // Load the input argument.
  // This should really be generated code (like module_abi_cc.h).
  int32_t arg0 = regs.i32[arg_list->registers[0] & regs.i32_mask];

  int32_t out0;

  add_module_add_call_impl(arg0, &out0);

  // Store the result.
  regs.i32[ret_list->registers[0] & regs.i32_mask] = out0;

  return iree_ok_status();
}

typedef iree_status_t (*add_module_func_t)(
    add_module_t* module, add_module_state_t* state, iree_vm_stack_t* stack,
    const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result);
static const add_module_func_t add_module_funcs_[] = {
    add_module_add_1,
    add_module_call_function,
};
static_assert(IREE_ARRAYSIZE(add_module_funcs_) ==
                  IREE_ARRAYSIZE(add_module_exports_),
              "function pointer table must be 1:1 with exports");

static iree_status_t IREE_API_PTR add_module_begin_call(
    void* self, iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result) {
  // NOTE: we aren't using module state in this module.
  return add_module_funcs_[call->function.ordinal](
      /*module=*/NULL, /*module_state=*/NULL, stack, call, out_result);
}

static iree_status_t add_module_create(iree_allocator_t allocator,
                                       iree_vm_module_t** out_module) {
  // NOTE: this module has neither shared or per-context module state.
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  interface.begin_call = add_module_begin_call;
  return iree_vm_native_module_create(&interface, &add_module_descriptor_,
                                      allocator, out_module);
}
