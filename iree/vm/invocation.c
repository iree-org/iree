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

#include "iree/base/api.h"
#include "iree/base/tracing.h"

#define VMMAX(a, b) (((a) > (b)) ? (a) : (b))
#define VMMIN(a, b) (((a) < (b)) ? (a) : (b))

// Marshals a variant list of values into callee registers.
// The |out_dst_reg_list| will be populated with the register ordinals and must
// be preallocated to store iree_vm_variant_list_size inputs.
static void iree_vm_stack_frame_marshal_inputs(
    iree_vm_variant_list_t* inputs, const iree_vm_registers_t dst_regs,
    iree_vm_register_list_t* out_dst_reg_list) {
  iree_host_size_t count = iree_vm_variant_list_size(inputs);
  uint16_t i32_reg = 0;
  uint16_t ref_reg = 0;
  out_dst_reg_list->size = (uint16_t)count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_vm_variant_t* variant = iree_vm_variant_list_get(inputs, i);
    if (IREE_VM_VARIANT_IS_REF(variant)) {
      out_dst_reg_list->registers[i] =
          ref_reg | IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT;
      iree_vm_ref_t* reg_ref = &dst_regs.ref[ref_reg++];
      memset(reg_ref, 0, sizeof(*reg_ref));
      iree_vm_ref_retain(&variant->ref, reg_ref);
    } else {
      out_dst_reg_list->registers[i] = i32_reg;
      dst_regs.i32[i32_reg++] = variant->i32;
    }
  }
}

// Marshals callee return registers into a variant list.
static iree_status_t iree_vm_stack_frame_marshal_outputs(
    const iree_vm_registers_t src_regs,
    const iree_vm_register_list_t* src_reg_list,
    iree_vm_variant_list_t* outputs) {
  for (int i = 0; i < src_reg_list->size; ++i) {
    uint16_t reg = src_reg_list->registers[i];
    if (reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_t* value = &src_regs.ref[reg & src_regs.ref_mask];
      IREE_RETURN_IF_ERROR(
          iree_vm_variant_list_append_ref_move(outputs, value));
    } else {
      iree_vm_value_t value;
      value.type = IREE_VM_VALUE_TYPE_I32;
      value.i32 = src_regs.i32[reg & src_regs.i32_mask];
      IREE_RETURN_IF_ERROR(iree_vm_variant_list_append_value(outputs, value));
    }
  }
  return IREE_STATUS_OK;
}

// TODO(benvanik): implement this as an iree_vm_invocation_t sequence.
static iree_status_t iree_vm_invoke_within(
    iree_vm_context_t* context, iree_vm_stack_t* stack,
    iree_vm_function_t function, const iree_vm_invocation_policy_t* policy,
    iree_vm_variant_list_t* inputs, iree_vm_variant_list_t* outputs) {
  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  // TODO(#2075): disabled because check_test is invoking native methods.
  // These checks should be nice and simple as we don't support variadic
  // args/results in bytecode.
  int32_t input_count = inputs ? iree_vm_variant_list_size(inputs) : 0;
  int32_t min_output_count = VMMAX(signature.result_count, outputs ? 1 : 0);
  // if (input_count != signature.argument_count) {
  //   return IREE_STATUS_INVALID_ARGUMENT;
  // } else if (!outputs && signature.result_count > 0) {
  //   return IREE_STATUS_INVALID_ARGUMENT;
  // }

  // Allocate storage for marshaling arguments into the callee stack frame.
  iree_vm_register_list_t* argument_registers =
      (iree_vm_register_list_t*)iree_alloca(sizeof(iree_vm_register_list_t) +
                                            input_count * sizeof(uint16_t));
  argument_registers->size = input_count;
  iree_vm_register_list_t* result_registers =
      (iree_vm_register_list_t*)iree_alloca(sizeof(iree_vm_register_list_t) +
                                            min_output_count *
                                                sizeof(uint16_t));
  result_registers->size = min_output_count;

  // Enter the [external] frame, which will have storage space for the
  // argument and result registers.
  int32_t register_count = VMMAX(input_count, min_output_count);
  iree_vm_stack_frame_t* external_frame = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_stack_external_enter(stack, iree_make_cstring_view("invoke"),
                                   register_count, &external_frame));

  // Marshal inputs into the external stack frame registers.
  if (inputs) {
    iree_vm_stack_frame_marshal_inputs(inputs, external_frame->registers,
                                       argument_registers);
  }

  // Perform execution. Note that for synchronous execution we expect this to
  // complete without yielding.
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.argument_registers = argument_registers;
  call.result_registers = result_registers;
  iree_vm_execution_result_t result;
  IREE_RETURN_IF_ERROR(function.module->begin_call(function.module->self, stack,
                                                   &call, &result));

  // Read back the outputs from the [external] marshaling stack frame.
  external_frame = iree_vm_stack_current_frame(stack);
  if (outputs) {
    IREE_RETURN_IF_ERROR(iree_vm_stack_frame_marshal_outputs(
        external_frame->registers, result_registers, outputs));
  }

  iree_vm_stack_external_leave(stack);

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_invoke(
    iree_vm_context_t* context, iree_vm_function_t function,
    const iree_vm_invocation_policy_t* policy, iree_vm_variant_list_t* inputs,
    iree_vm_variant_list_t* outputs, iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate a VM stack on the host stack and initialize it.
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack, iree_vm_context_state_resolver(context), allocator);
  iree_status_t status =
      iree_vm_invoke_within(context, stack, function, policy, inputs, outputs);
  iree_vm_stack_deinitialize(stack);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
