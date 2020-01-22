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

#include "iree/vm/invocation.h"

static iree_status_t iree_vm_validate_function_inputs(
    iree_vm_function_t function, iree_vm_variant_list_t* inputs) {
  // TODO(benvanik): validate inputs.
  return IREE_STATUS_OK;
}

static iree_status_t iree_vm_marshal_inputs(
    iree_vm_variant_list_t* inputs, iree_vm_stack_frame_t* callee_frame) {
  iree_vm_registers_t* registers = &callee_frame->registers;
  iree_host_size_t count = iree_vm_variant_list_size(inputs);
  int i32_reg = 0;
  int ref_reg = 0;
  for (int i = 0; i < count; ++i) {
    iree_vm_variant_t* variant = iree_vm_variant_list_get(inputs, i);
    if (IREE_VM_VARIANT_IS_REF(variant)) {
      iree_vm_ref_t* reg_ref = &registers->ref[ref_reg++];
      memset(reg_ref, 0, sizeof(*reg_ref));
      iree_vm_ref_retain(&variant->ref, reg_ref);
    } else {
      registers->i32[i32_reg++] = variant->i32;
    }
  }
  registers->ref_register_count = ref_reg;
  return IREE_STATUS_OK;
}

static iree_status_t iree_vm_marshal_outputs(
    iree_vm_stack_frame_t* callee_frame, iree_vm_variant_list_t* outputs) {
  iree_vm_registers_t* registers = &callee_frame->registers;
  const iree_vm_register_list_t* return_registers =
      callee_frame->return_registers;
  for (int i = 0; i < return_registers->size; ++i) {
    uint8_t reg = return_registers->registers[i];
    if (reg & IREE_REF_REGISTER_TYPE_BIT) {
      // Always move (as the stack frame will be destroyed soon).
      IREE_RETURN_IF_ERROR(iree_vm_variant_list_append_ref_move(
          outputs, &registers->ref[reg & IREE_REF_REGISTER_MASK]));
    } else {
      iree_vm_value_t value;
      value.type = IREE_VM_VALUE_TYPE_I32;
      value.i32 = registers->i32[reg & IREE_I32_REGISTER_MASK];
      IREE_RETURN_IF_ERROR(iree_vm_variant_list_append_value(outputs, value));
    }
  }
  return IREE_STATUS_OK;
}

// TODO(benvanik): implement this as an iree_vm_invocation_t sequence.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_invoke(
    iree_vm_context_t* context, iree_vm_function_t function,
    const iree_vm_invocation_policy_t* policy, iree_vm_variant_list_t* inputs,
    iree_vm_variant_list_t* outputs, iree_allocator_t allocator) {
  // NOTE: it is ok to have no inputs or outputs. If we do have them, though,
  // they must be valid.
  // TODO(benvanik): validate outputs capacity.
  IREE_RETURN_IF_ERROR(iree_vm_validate_function_inputs(function, inputs));

  // Allocate a stack on the heap and initialize it.
  // If we shrunk the stack (or made it so that it could dynamically grow)
  // then we could stack-allocate it here and not need the allocator at all.
  iree_vm_stack_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(iree_vm_stack_t),
                                             (void**)&stack));
  iree_status_t status =
      iree_vm_stack_init(iree_vm_context_state_resolver(context), stack);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, stack);
    return status;
  }

  iree_vm_stack_frame_t* callee_frame = NULL;
  status = iree_vm_stack_function_enter(stack, function, &callee_frame);

  // Marshal inputs.
  if (iree_status_is_ok(status) && inputs) {
    status = iree_vm_marshal_inputs(inputs, callee_frame);
  }

  // Perform execution. Note that for synchronous execution we expect this to
  // complete without yielding.
  if (iree_status_is_ok(status)) {
    iree_vm_execution_result_t result;
    status = function.module->execute(function.module->self, stack,
                                      callee_frame, &result);
  }

  // Marshal outputs.
  if (iree_status_is_ok(status) && outputs) {
    status = iree_vm_marshal_outputs(callee_frame, outputs);
  }

  iree_vm_stack_function_leave(stack);
  iree_vm_stack_deinit(stack);
  iree_allocator_free(allocator, stack);
  return status;
}
