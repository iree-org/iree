// Copyright 2019 Google LLC
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

#include "iree/vm2/stack.h"

#include <string.h>

#include "iree/vm2/module.h"

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_init(
    iree_vm_state_resolver_t state_resolver, iree_vm_stack_t* out_stack) {
  memset(out_stack, 0, sizeof(iree_vm_stack_t));
  out_stack->state_resolver = state_resolver;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_deinit(iree_vm_stack_t* stack) {
  while (stack->depth) {
    IREE_API_RETURN_IF_API_ERROR(iree_vm_stack_function_leave(stack));
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_current_frame(iree_vm_stack_t* stack) {
  return stack->depth > 0 ? &stack->frames[stack->depth - 1] : NULL;
}

IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_parent_frame(iree_vm_stack_t* stack) {
  return stack->depth > 1 ? &stack->frames[stack->depth - 2] : NULL;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_function_enter(
    iree_vm_stack_t* stack, iree_vm_function_t function,
    iree_vm_stack_frame_t** out_callee_frame) {
  *out_callee_frame = NULL;
  if (stack->depth == IREE_MAX_STACK_DEPTH) {
    return IREE_STATUS_RESOURCE_EXHAUSTED;
  }

  // Try to reuse the same module state if the caller and callee are from the
  // same module. Otherwise, query the state from the registered handler.
  iree_vm_stack_frame_t* callee_frame = &stack->frames[stack->depth];
  if (stack->depth > 0) {
    iree_vm_stack_frame_t* caller_frame = &stack->frames[stack->depth - 1];
    if (caller_frame->function.module == function.module) {
      callee_frame->module_state = caller_frame->module_state;
    }
  }
  if (!callee_frame->module_state) {
    IREE_API_RETURN_IF_API_ERROR(stack->state_resolver.query_module_state(
        stack->state_resolver.self, function.module,
        &callee_frame->module_state));
  }

  ++stack->depth;

  callee_frame->function = function;
  callee_frame->offset = 0;
  callee_frame->registers.ref_register_count = 0;
  callee_frame->return_registers = NULL;

#ifndef NDEBUG
  memset(callee_frame->registers.i32, 0xCD,
         sizeof(callee_frame->registers.i32));
  memset(callee_frame->registers.ref, 0xCD,
         sizeof(callee_frame->registers.ref));
#endif  // !NDEBUG

  *out_callee_frame = callee_frame;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_function_leave(iree_vm_stack_t* stack) {
  if (stack->depth <= 0) {
    return IREE_STATUS_FAILED_PRECONDITION;
  }

  iree_vm_stack_frame_t* callee_frame = &stack->frames[stack->depth - 1];
  --stack->depth;
  memset(&callee_frame->function, 0, sizeof(callee_frame->function));
  callee_frame->module_state = NULL;

  iree_vm_registers_t* registers = &callee_frame->registers;
  for (int i = 0; i < registers->ref_register_count; ++i) {
    iree_vm_ref_release(&registers->ref[i]);
  }

  return IREE_STATUS_OK;
}
