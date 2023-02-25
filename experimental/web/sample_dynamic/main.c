// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/bytecode/module.h"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Opaque state for the sample, shared between multiple loaded programs.
typedef struct iree_sample_state_t iree_sample_state_t;

// Initializes the sample and returns its state.
iree_sample_state_t* setup_sample();

// Shuts down the sample and frees its state.
// Requires that all programs first be unloaded with |unload_program|.
void cleanup_sample(iree_sample_state_t* sample_state);

// Opaque state for an individual loaded program.
typedef struct iree_program_state_t iree_program_state_t;

// Loads a program into the sample from the provided data.
// Note: this takes ownership of |vmfb_data|.
iree_program_state_t* load_program(iree_sample_state_t* sample_state,
                                   uint8_t* vmfb_data, size_t length);

// Inspects metadata about a loaded program, printing to stdout.
void inspect_program(iree_program_state_t* program_state);

// Unloads a program and frees its state.
void unload_program(iree_program_state_t* program_state);

// Calls a function synchronously.
//
// Returns a semicolon-delimited list of formatted outputs on success or the
// empty string on failure. Note: This is in need of some real API bindings
// that marshal structured data between C <-> JS.
//
// * |function_name| is the fully qualified function name, like 'module.abs'.
// * |inputs| is a semicolon delimited list of VM scalars and buffers, as
//   described in iree/tooling/vm_util and used in IREE's CLI tools.
//   For example, the CLI `--input=f32=1 --input=f32=2`
//   should be passed here as `f32=1;f32=2`.
// * |iterations| is the number of times to call the function, for benchmarking
const char* call_function(iree_program_state_t* program_state,
                          const char* function_name, const char* inputs,
                          int iterations);

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_sample_state_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
} iree_sample_state_t;

typedef struct iree_program_state_t {
  iree_runtime_session_t* session;
  iree_vm_module_t* module;
} iree_program_state_t;

extern iree_status_t create_device_with_loaders(iree_allocator_t host_allocator,
                                                iree_hal_device_t** out_device);

iree_sample_state_t* setup_sample() {
  iree_sample_state_t* sample_state = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_allocator_system(),
                            sizeof(iree_sample_state_t), (void**)&sample_state);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().

  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &sample_state->instance);
  }

  if (iree_status_is_ok(status)) {
    status = create_device_with_loaders(iree_allocator_system(),
                                        &sample_state->device);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    cleanup_sample(sample_state);
    return NULL;
  }

  return sample_state;
}

void cleanup_sample(iree_sample_state_t* sample_state) {
  iree_hal_device_release(sample_state->device);
  iree_runtime_instance_release(sample_state->instance);
  free(sample_state);
}

iree_program_state_t* load_program(iree_sample_state_t* sample_state,
                                   uint8_t* vmfb_data, size_t length) {
  iree_program_state_t* program_state = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(iree_program_state_t),
                                               (void**)&program_state);

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        sample_state->instance, &session_options, sample_state->device,
        iree_runtime_instance_host_allocator(sample_state->instance),
        &program_state->session);
  }

  if (iree_status_is_ok(status)) {
    // Take ownership of the FlatBuffer data so JavaScript doesn't need to
    // explicitly call `Module._free()`.
    status = iree_vm_bytecode_module_create(
        iree_runtime_instance_vm_instance(sample_state->instance),
        iree_make_const_byte_span(vmfb_data, length),
        /*flatbuffer_allocator=*/iree_allocator_system(),
        iree_allocator_system(), &program_state->module);
  } else {
    // Must clean up the FlatBuffer data directly.
    iree_allocator_free(iree_allocator_system(), (void*)vmfb_data);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(program_state->session,
                                                program_state->module);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    unload_program(program_state);
    return NULL;
  }

  return program_state;
}

void inspect_program(iree_program_state_t* program_state) {
  fprintf(stdout, "=== program properties ===\n");

  iree_vm_module_t* module = program_state->module;
  iree_string_view_t module_name = iree_vm_module_name(module);
  fprintf(stdout, "  module name: '%.*s'\n", (int)module_name.size,
          module_name.data);

  iree_vm_module_signature_t module_signature =
      iree_vm_module_signature(module);
  fprintf(stdout, "  module signature:\n");
  fprintf(stdout, "    %" PRIhsz " imported functions\n",
          module_signature.import_function_count);
  fprintf(stdout, "    %" PRIhsz " exported functions\n",
          module_signature.export_function_count);
  fprintf(stdout, "    %" PRIhsz " internal functions\n",
          module_signature.internal_function_count);

  fprintf(stdout, "  exported functions:\n");
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    iree_vm_function_t function;
    iree_status_t status = iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      continue;
    }

    iree_string_view_t function_name = iree_vm_function_name(&function);
    iree_vm_function_signature_t function_signature =
        iree_vm_function_signature(&function);
    iree_string_view_t calling_convention =
        function_signature.calling_convention;
    fprintf(stdout, "    function name: '%.*s', calling convention: %.*s'\n",
            (int)function_name.size, function_name.data,
            (int)calling_convention.size, calling_convention.data);
  }
}

void unload_program(iree_program_state_t* program_state) {
  iree_vm_module_release(program_state->module);
  iree_runtime_session_release(program_state->session);
  free(program_state);
}

static iree_status_t parse_input_into_call(
    iree_runtime_call_t* call, iree_hal_allocator_t* device_allocator,
    iree_string_view_t input) {
  bool has_equal =
      iree_string_view_find_char(input, '=', 0) != IREE_STRING_VIEW_NPOS;
  bool has_x =
      iree_string_view_find_char(input, 'x', 0) != IREE_STRING_VIEW_NPOS;
  if (has_equal || has_x) {
    // Buffer view (either just a shape or a shape=value) or buffer.
    bool is_storage_reference =
        iree_string_view_consume_prefix(&input, iree_make_cstring_view("&"));
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_parse(input, device_allocator, &buffer_view),
        "parsing value '%.*s'", (int)input.size, input.data);
    if (is_storage_reference) {
      // Storage buffer reference; just take the storage for the buffer view -
      // it'll still have whatever contents were specified (or 0) but we'll
      // discard the metadata.
      iree_vm_ref_t buffer_ref =
          iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view));
      iree_hal_buffer_view_release(buffer_view);
      return iree_vm_list_push_ref_move(call->inputs, &buffer_ref);
    } else {
      iree_vm_ref_t buffer_view_ref =
          iree_hal_buffer_view_move_ref(buffer_view);
      return iree_vm_list_push_ref_move(call->inputs, &buffer_view_ref);
    }
  } else {
    // Scalar.
    bool has_dot =
        iree_string_view_find_char(input, '.', 0) != IREE_STRING_VIEW_NPOS;
    iree_vm_value_t val;
    if (has_dot) {
      // Float.
      val = iree_vm_value_make_f32(0.0f);
      if (!iree_string_view_atof(input, &val.f32)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parsing value '%.*s' as f32", (int)input.size,
                                input.data);
      }
    } else {
      // Integer.
      val = iree_vm_value_make_i32(0);
      if (!iree_string_view_atoi_int32(input, &val.i32)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parsing value '%.*s' as i32", (int)input.size,
                                input.data);
      }
    }
    return iree_vm_list_push_value(call->inputs, &val);
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "Unhandled function input (unreachable?)");
}

static iree_status_t parse_inputs_into_call(
    iree_runtime_call_t* call, iree_hal_allocator_t* device_allocator,
    iree_string_view_t inputs) {
  if (inputs.size == 0) return iree_ok_status();

  // Inputs are provided in a semicolon-delimited list.
  // Split inputs from the list until no semicolons are left.
  iree_string_view_t remaining_inputs = inputs;
  intptr_t split_index = 0;
  do {
    iree_string_view_t next_input;
    split_index = iree_string_view_split(remaining_inputs, ';', &next_input,
                                         &remaining_inputs);
    IREE_RETURN_IF_ERROR(
        parse_input_into_call(call, device_allocator, next_input));
  } while (split_index != -1);

  return iree_ok_status();
}

static iree_status_t print_outputs_from_call(
    iree_runtime_call_t* call, iree_string_builder_t* outputs_builder) {
  iree_vm_list_t* variants_list = iree_runtime_call_outputs(call);
  for (iree_host_size_t i = 0; i < iree_vm_list_size(variants_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(variants_list, i, &variant),
                         "variant %" PRIhsz " not present", i);

    if (iree_vm_variant_is_value(variant)) {
      switch (variant.type.value_type) {
        case IREE_VM_VALUE_TYPE_I8: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i8=%" PRIi8, variant.i8));
          break;
        }
        case IREE_VM_VALUE_TYPE_I16: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i16=%" PRIi16, variant.i16));
          break;
        }
        case IREE_VM_VALUE_TYPE_I32: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i32=%" PRIi32, variant.i32));
          break;
        }
        case IREE_VM_VALUE_TYPE_I64: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "i64=%" PRIi64, variant.i64));
          break;
        }
        case IREE_VM_VALUE_TYPE_F32: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "f32=%f", variant.f32));
          break;
        }
        case IREE_VM_VALUE_TYPE_F64: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              outputs_builder, "f64=%lf", variant.f64));
          break;
        }
        default: {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(outputs_builder, "?"));
          break;
        }
      }
    } else if (iree_vm_variant_is_ref(variant)) {
      if (iree_hal_buffer_view_isa(variant.ref)) {
        iree_hal_buffer_view_t* buffer_view =
            iree_hal_buffer_view_deref(variant.ref);

        // Query total length (excluding NUL terminator).
        iree_host_size_t result_length = 0;
        iree_status_t status = iree_hal_buffer_view_format(
            buffer_view, SIZE_MAX, 0, NULL, &result_length);
        if (!iree_status_is_out_of_range(status)) return status;
        ++result_length;  // include NUL

        // Allocate scratch heap memory for the result and format into it.
        char* result_str = NULL;
        IREE_RETURN_IF_ERROR(iree_allocator_malloc(
            iree_allocator_system(), result_length, (void**)&result_str));
        IREE_RETURN_IF_ERROR(iree_hal_buffer_view_format(
            buffer_view, SIZE_MAX, result_length, result_str, &result_length));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            outputs_builder, "%.*s", (int)result_length, result_str));
        iree_allocator_free(iree_allocator_system(), result_str);
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            outputs_builder, "(no printer)"));
      }
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(outputs_builder, "(null)"));
    }

    if (i < iree_vm_list_size(variants_list) - 1) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(outputs_builder, ";"));
    }
  }

  iree_vm_list_resize(variants_list, 0);

  return iree_ok_status();
}

const char* call_function(iree_program_state_t* program_state,
                          const char* function_name, const char* inputs,
                          int iterations) {
  iree_status_t status = iree_ok_status();

  // Fully qualify the function name. This sample only supports loading one
  // module (i.e. 'program') per session, so we can do this.
  iree_string_builder_t name_builder;
  iree_string_builder_initialize(iree_allocator_system(), &name_builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t module_name = iree_vm_module_name(program_state->module);
    status = iree_string_builder_append_format(&name_builder, "%.*s.%s",
                                               (int)module_name.size,
                                               module_name.data, function_name);
  }

  iree_runtime_call_t call;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        program_state->session, iree_string_builder_view(&name_builder), &call);
  }
  iree_string_builder_deinitialize(&name_builder);

  if (iree_status_is_ok(status)) {
    status = parse_inputs_into_call(
        &call, iree_runtime_session_device_allocator(program_state->session),
        iree_make_cstring_view(inputs));
  }

  // Note: Timing has ~millisecond precision on the web to mitigate timing /
  // side-channel security threats.
  // https://developer.mozilla.org/en-US/docs/Web/API/Performance/now#reduced_time_precision
  iree_time_t start_time = iree_time_now();
  for (int i = 0; i < iterations; ++i) {
    if (iree_status_is_ok(status)) {
      status = iree_runtime_call_invoke(&call, /*flags=*/0);
    }
  }
  iree_time_t end_time = iree_time_now();
  iree_time_t time_elapsed = end_time - start_time;

  iree_string_builder_t outputs_builder;
  iree_string_builder_initialize(iree_allocator_system(), &outputs_builder);

  // Output a JSON object as a string:
  // {
  //   "total_invoke_time_ms": [number],
  //   "outputs": [semicolon delimited list of formatted outputs]
  // }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &outputs_builder,
        "{ \"total_invoke_time_ms\": %" PRId64 ", \"outputs\": \"",
        time_elapsed / 1000000);
  }
  if (iree_status_is_ok(status)) {
    status = print_outputs_from_call(&call, &outputs_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&outputs_builder, "\"}");
  }

  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&outputs_builder);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return "";
  }

  // Note: this leaks the buffer. It's up to the caller to free it after use.
  return iree_string_builder_buffer(&outputs_builder);
}
