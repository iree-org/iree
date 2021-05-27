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

// Marshals caller arguments from the variant list to the ABI convention.
static iree_status_t iree_vm_invoke_marshal_inputs(
    iree_string_view_t cconv_arguments, iree_vm_list_t* inputs,
    iree_byte_span_t arguments) {
  // We are 1:1 right now with no variadic args, so do a quick verification on
  // the input list.
  iree_host_size_t expected_input_count =
      cconv_arguments.size > 0
          ? (cconv_arguments.data[0] == 'v' ? 0 : cconv_arguments.size)
          : 0;
  if (IREE_UNLIKELY(!inputs)) {
    if (IREE_UNLIKELY(expected_input_count > 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no input provided to a function that has inputs");
    }
    return iree_ok_status();
  } else if (IREE_UNLIKELY(expected_input_count != iree_vm_list_size(inputs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input list and function mismatch; expected %zu "
                            "arguments but passed %zu",
                            expected_input_count, iree_vm_list_size(inputs));
  }

  uint8_t* p = arguments.data;
  for (iree_host_size_t cconv_i = 0, arg_i = 0; cconv_i < cconv_arguments.size;
       ++cconv_i, ++arg_i) {
    switch (cconv_arguments.data[cconv_i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_I32, &value));
        memcpy(p, &value.i32, sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_I64, &value));
        memcpy(p, &value.i64, sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_F32: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_F32, &value));
        memcpy(p, &value.f32, sizeof(float));
        p += sizeof(float);
      } break;
      case IREE_VM_CCONV_TYPE_F64: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_F64, &value));
        memcpy(p, &value.f64, sizeof(double));
        p += sizeof(double);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        // TODO(benvanik): see if we can't remove this retain by instead relying
        // on the caller still owning the list.
        IREE_RETURN_IF_ERROR(
            iree_vm_list_get_ref_retain(inputs, arg_i, (iree_vm_ref_t*)p));
        p += sizeof(iree_vm_ref_t);
      } break;
    }
  }
  return iree_ok_status();
}

// Marshals callee results from the ABI convention to the variant list.
static iree_status_t iree_vm_invoke_marshal_outputs(
    iree_string_view_t cconv_results, iree_byte_span_t results,
    iree_vm_list_t* outputs) {
  iree_host_size_t expected_output_count =
      cconv_results.size > 0
          ? (cconv_results.data[0] == 'v' ? 0 : cconv_results.size)
          : 0;
  if (IREE_UNLIKELY(!outputs)) {
    if (IREE_UNLIKELY(expected_output_count > 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no output provided to a function that has outputs");
    }
    return iree_ok_status();
  }

  // Resize the output list to hold all results (and kill anything that may
  // have been in there).
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(outputs, 0));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(outputs, expected_output_count));

  uint8_t* p = results.data;
  for (iree_host_size_t cconv_i = 0, arg_i = 0; cconv_i < cconv_results.size;
       ++cconv_i, ++arg_i) {
    switch (cconv_results.data[cconv_i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32: {
        iree_vm_value_t value = iree_vm_value_make_i32(*(int32_t*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64: {
        iree_vm_value_t value = iree_vm_value_make_i64(*(int64_t*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_F32: {
        iree_vm_value_t value = iree_vm_value_make_f32(*(float*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(float);
      } break;
      case IREE_VM_CCONV_TYPE_F64: {
        iree_vm_value_t value = iree_vm_value_make_f64(*(double*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(double);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        IREE_RETURN_IF_ERROR(
            iree_vm_list_set_ref_move(outputs, arg_i, (iree_vm_ref_t*)p));
        p += sizeof(iree_vm_ref_t);
      } break;
    }
  }
  return iree_ok_status();
}

// TODO(benvanik): implement this as an iree_vm_invocation_t sequence.
static iree_status_t iree_vm_invoke_within(
    iree_vm_context_t* context, iree_vm_stack_t* stack,
    iree_vm_function_t function, const iree_vm_invocation_policy_t* policy,
    iree_vm_list_t* inputs, iree_vm_list_t* outputs) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stack);

  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t cconv_arguments = iree_string_view_empty();
  iree_string_view_t cconv_results = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      &signature, &cconv_arguments, &cconv_results));

  // Marshal the input arguments into the VM ABI and preallocate the result
  // buffer.
  // NOTE: today we don't support variadic arguments through this interface.
  iree_byte_span_t arguments = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(iree_vm_function_call_compute_cconv_fragment_size(
      cconv_arguments, /*segment_size_list=*/NULL, &arguments.data_length));
  arguments.data = iree_alloca(arguments.data_length);
  memset(arguments.data, 0, arguments.data_length);
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke_marshal_inputs(cconv_arguments, inputs, arguments));

  // Allocate the result output that will be populated by the callee.
  iree_byte_span_t results = iree_make_byte_span(NULL, 0);
  IREE_RETURN_IF_ERROR(iree_vm_function_call_compute_cconv_fragment_size(
      cconv_results, /*segment_size_list=*/NULL, &results.data_length));
  results.data = iree_alloca(results.data_length);
  memset(results.data, 0, results.data_length);

  // Perform execution. Note that for synchronous execution we expect this to
  // complete without yielding.
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = arguments;
  call.results = results;
  iree_vm_execution_result_t result;
  iree_status_t status =
      function.module->begin_call(function.module->self, stack, &call, &result);
  if (!iree_status_is_ok(status)) {
    iree_vm_function_call_release(&call, &signature);
    return status;
  }

  // Read back the outputs from the result buffer.
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke_marshal_outputs(cconv_results, results, outputs));

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_invoke(
    iree_vm_context_t* context, iree_vm_function_t function,
    const iree_vm_invocation_policy_t* policy, iree_vm_list_t* inputs,
    iree_vm_list_t* outputs, iree_allocator_t allocator) {
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
