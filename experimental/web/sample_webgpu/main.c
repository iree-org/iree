// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Must be first.
#include "experimental/webgpu/platform/webgpu.h"

// NOTE: include order matters.
#include "experimental/webgpu/buffer.h"
#include "experimental/webgpu/webgpu_device.h"
#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/loop_emscripten.h"
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

// Callback function passed in from JS.
// On call failure, is called with the empty string.
// On call success, is called with a JSON object:
//   {
//     "invoke_time_ms": [number],
//     "readback_time_ms": [number],
//     "outputs": [semicolon delimited list of formatted outputs]
//   }
// TODO(scotttodd): on error, call with some other JSON / iree_status_fprint
// TODO(scotttodd): on success, return structured data instead of formatted text
typedef void(IREE_API_PTR* iree_call_function_callback_fn_t)(char* output);

// Calls a function asynchronously.
//
// Returns true if call setup was successful, or false if setup failed.
//
// * |function_name| is the function name, like 'abs' (not fully qualified).
// * |inputs| is a semicolon delimited list of VM scalars and buffers, as
//   described in iree/tooling/vm_util and used in IREE's CLI tools.
//   For example, the CLI `--function_input=f32=1 --function_input=f32=2`
//   should be passed here as `f32=1;f32=2`.
// * |completion_callback_fn| will be called after the call completes.
const bool call_function(
    iree_program_state_t* program_state, const char* function_name,
    const char* inputs,
    iree_call_function_callback_fn_t completion_callback_fn);

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

typedef struct iree_sample_state_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_loop_emscripten_t* loop;
} iree_sample_state_t;

typedef struct iree_program_state_t {
  iree_sample_state_t* sample_state;
  iree_runtime_session_t* session;
  iree_vm_module_t* module;
} iree_program_state_t;

// Function calls run asynchronously, both for the computations themselves
// and for mapping the outputs back from GPU/device memory to CPU/host memory.
//
// Here is the high level flow:
//
//   call_function
//     parse_inputs_into_call
//     iree_vm_async_invoke
//       [async] invoke_callback
//         process_call_outputs
//           <batch transfer from output buffers to mappable device buffers>
//           map_call_output[0...n]
//             wgpuBufferMapAsync
//               [async] buffer_map_async_callback
//                 <set event>
//           [async] iree_loop_wait_all
//             map_all_callback
//               print_outputs_from_call
//               issue top level callback
//
// State tracking uses the iree_call_function_state_t and iree_output_state_t
// structs defined below, keeping both live until the final map_all_callback.
// If any of the asynchronous calls fail, all other calls must be treated as
// failed and all state must be freed.

// State for a single output within an asynchronous function call.
//
// If the output is a buffer, optional fields are used to track asynchronous
// mapping from device memory to host memory.
typedef struct iree_output_state_t {
  // Event that will be signaled when the output is ready to access on the host.
  // * For non-buffer outputs, this will start signaled.
  // * For buffer outputs, this will be signaled when _either_
  //   |mapped_host_buffer| points to the output data _or_ mapping failed.
  iree_event_t ready_event;

  // The original buffer_view from the call output, if the output is a buffer.
  // Not guaranteed to reference mappable memory.
  iree_hal_buffer_view_t* buffer_view;

  // A mappable device buffer (backed by a WGPUBuffer).
  // The original buffer will be copied into this buffer by a transfer command.
  iree_hal_buffer_t* mappable_device_buffer;

  // A mapped host buffer (backed by a iree_hal_heap_buffer_t).
  // After asynchronous mapping of |mappable_device_buffer|, the output data
  // will be wrapped into this buffer.
  // Note: to recover the original shape, use iree_hal_buffer_view_create_like.
  iree_hal_buffer_t* mapped_host_buffer;
} iree_output_state_t;

// Aggregate state for an asynchronous function call.
typedef struct iree_call_function_state_t {
  iree_runtime_call_t call;
  iree_loop_emscripten_t* loop;
  iree_call_function_callback_fn_t callback_fn;

  // Opaque state used by iree_vm_async_invoke.
  iree_vm_async_invoke_state_t* invoke_state;

  // Timing/statistics metadata (~millisecond precision on the web).
  // https://developer.mozilla.org/en-US/docs/Web/API/Performance/now#reduced_time_precision
  iree_time_t invoke_start_time;
  iree_time_t invoke_end_time;
  iree_time_t readback_start_time;
  iree_time_t readback_end_time;

  // Sticky status for the first async error.
  // If this is not ok, treat all output buffer mappings as having errored and
  // issue the callback with a failure message.
  iree_status_t async_status;

  // Output processing state.
  iree_host_size_t outputs_size;
  iree_output_state_t* output_states;
} iree_call_function_state_t;

static void iree_call_function_state_destroy(
    iree_call_function_state_t* call_state) {
  // Output processing state.
  for (iree_host_size_t i = 0; i < call_state->outputs_size; ++i) {
    if (call_state->output_states[i].buffer_view) {
      iree_hal_buffer_release(call_state->output_states[i].mapped_host_buffer);
      iree_hal_buffer_release(
          call_state->output_states[i].mappable_device_buffer);
      iree_hal_buffer_view_release(call_state->output_states[i].buffer_view);
    }
    iree_event_deinitialize(&call_state->output_states[i].ready_event);
  }
  iree_allocator_free(iree_allocator_system(), call_state->output_states);
  iree_status_free(call_state->async_status);

  // Invoke state.
  iree_allocator_free(iree_allocator_system(), call_state->invoke_state);
  iree_runtime_call_deinitialize(&call_state->call);

  iree_allocator_free(iree_allocator_system(), call_state);
}

extern iree_status_t create_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device);

iree_sample_state_t* setup_sample() {
  iree_sample_state_t* sample_state = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*sample_state), (void**)&sample_state);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().

  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &sample_state->instance);
  }

  if (iree_status_is_ok(status)) {
    status = create_device(iree_allocator_system(), &sample_state->device);
  }

  if (iree_status_is_ok(status)) {
    status = iree_loop_emscripten_allocate(iree_allocator_system(),
                                           &sample_state->loop);
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
  iree_loop_emscripten_free(sample_state->loop);
  iree_hal_device_release(sample_state->device);
  iree_runtime_instance_release(sample_state->instance);
  free(sample_state);
}

iree_program_state_t* load_program(iree_sample_state_t* sample_state,
                                   uint8_t* vmfb_data, size_t length) {
  iree_program_state_t* program_state = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*program_state), (void**)&program_state);
  program_state->sample_state = sample_state;

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

//===----------------------------------------------------------------------===//
// Input parsing
//===----------------------------------------------------------------------===//

static iree_status_t parse_input_into_call(
    iree_runtime_call_t* call, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_string_view_t input) {
  bool has_equal =
      iree_string_view_find_char(input, '=', 0) != IREE_STRING_VIEW_NPOS;
  bool has_x =
      iree_string_view_find_char(input, 'x', 0) != IREE_STRING_VIEW_NPOS;
  if (has_equal || has_x) {
    // Buffer view (either just a shape or a shape=value) or buffer.
    bool is_storage_reference =
        iree_string_view_consume_prefix(&input, iree_make_cstring_view("&"));
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
                             input, device, device_allocator, &buffer_view),
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
    iree_runtime_call_t* call, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_string_view_t inputs) {
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
        parse_input_into_call(call, device, device_allocator, next_input));
  } while (split_index != -1);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Output readback and formatting
//===----------------------------------------------------------------------===//

typedef struct iree_buffer_map_userdata_t {
  iree_call_function_state_t* call_state;
  iree_host_size_t buffer_index;
} iree_buffer_map_userdata_t;

static void iree_webgpu_mapped_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  WGPUBuffer buffer_handle = (WGPUBuffer)user_data;
  wgpuBufferUnmap(buffer_handle);
}

static void buffer_map_async_callback(WGPUBufferMapAsyncStatus map_status,
                                      void* userdata_ptr) {
  iree_status_t status = iree_ok_status();

  iree_buffer_map_userdata_t* userdata =
      (iree_buffer_map_userdata_t*)userdata_ptr;
  iree_host_size_t buffer_index = userdata->buffer_index;
  iree_hal_buffer_view_t* output_buffer_view =
      userdata->call_state->output_states[buffer_index].buffer_view;
  iree_hal_buffer_t* mappable_device_buffer =
      userdata->call_state->output_states[buffer_index].mappable_device_buffer;
  iree_hal_buffer_t** mapped_host_buffer_ptr =
      &userdata->call_state->output_states[buffer_index].mapped_host_buffer;

  switch (map_status) {
    case WGPUBufferMapAsyncStatus_Success:
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      fprintf(stderr, "  buffer_map_async_callback status: DeviceLost\n");
      break;
    default:
      fprintf(stderr, "  buffer_map_async_callback status: Error %d\n",
              map_status);
      break;
  }

  if (map_status != WGPUBufferMapAsyncStatus_Success) {
    status = iree_make_status(IREE_STATUS_UNKNOWN,
                              "wgpuBufferMapAsync failed for buffer %" PRIhsz,
                              buffer_index);
  }

  iree_device_size_t data_offset = 0;
  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(output_buffer_view);
  WGPUBuffer buffer_handle =
      iree_hal_webgpu_buffer_handle(mappable_device_buffer);

  // For this sample we want to print arbitrary buffers, which is easiest
  // using the |iree_hal_buffer_view_format| function. Internally, that
  // function requires synchronous buffer mapping, so we'll first wrap the
  // already (async) mapped GPU memory into a heap buffer. In a less general
  // application (or one not requiring pretty logging like this), we could
  // skip a few buffer copies and other data transformations here.

  const void* data_ptr = NULL;
  if (iree_status_is_ok(status)) {
    data_ptr =
        wgpuBufferGetConstMappedRange(buffer_handle, data_offset, data_length);
    if (data_ptr == NULL) {
      status = iree_make_status(
          IREE_STATUS_UNKNOWN,
          "wgpuBufferGetConstMappedRange failed for buffer %" PRIhsz,
          buffer_index);
    }
  }

  if (iree_status_is_ok(status)) {
    // The buffer we get from WebGPU may not be aligned to 64.
    iree_hal_memory_access_t memory_access =
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_UNALIGNED;
    status = iree_hal_heap_buffer_wrap(
        mappable_device_buffer->device_allocator,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL, memory_access,
        IREE_HAL_BUFFER_USAGE_MAPPING, data_length,
        iree_make_byte_span((void*)data_ptr, data_length),
        (iree_hal_buffer_release_callback_t){
            .fn = iree_webgpu_mapped_buffer_release,
            .user_data = buffer_handle,
        },
        mapped_host_buffer_ptr);
  }

  if (!iree_status_is_ok(status)) {
    // Set the sticky async error if not already set.
    userdata->call_state->async_status =
        iree_status_join(userdata->call_state->async_status, status);
  }

  iree_event_set(
      &userdata->call_state->output_states[buffer_index].ready_event);
  iree_allocator_free(iree_allocator_system(), userdata);
}

static iree_status_t print_outputs_from_call(
    iree_call_function_state_t* call_state,
    iree_string_builder_t* outputs_builder) {
  iree_vm_list_t* variants_list = iree_runtime_call_outputs(&call_state->call);
  for (iree_host_size_t i = 0; i < iree_vm_list_size(variants_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(variants_list, i, &variant),
        "variant %" PRIhsz " not present", i);

    if (iree_vm_variant_is_value(variant)) {
      switch (iree_vm_type_def_as_value(variant.type)) {
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
      // Interpret the mapped buffer in the same format as the output view.
      iree_hal_buffer_view_t* heap_buffer_view = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create_like(
          call_state->output_states[i].mapped_host_buffer,
          call_state->output_states[i].buffer_view, iree_allocator_system(),
          &heap_buffer_view));
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_append_to_builder(
          heap_buffer_view, SIZE_MAX, outputs_builder));
      iree_hal_buffer_view_release(heap_buffer_view);
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

static iree_status_t map_all_callback(void* user_data, iree_loop_t loop,
                                      iree_status_t status) {
  iree_call_function_state_t* call_state =
      (iree_call_function_state_t*)user_data;
  call_state->readback_end_time = iree_time_now();

  status = iree_status_join(call_state->async_status, status);

  iree_string_builder_t output_string_builder;
  iree_string_builder_initialize(iree_allocator_system(),
                                 &output_string_builder);

  // Output a JSON object as a string:
  // {
  //   "invoke_time_ms": [number],
  //   "readback_time_ms": [number],
  //   "outputs": [semicolon delimited list of formatted outputs]
  // }
  if (iree_status_is_ok(status)) {
    iree_time_t invoke_time_ms =
        (call_state->invoke_end_time - call_state->invoke_start_time) / 1000000;
    iree_time_t readback_time_ms =
        (call_state->readback_end_time - call_state->readback_start_time) /
        1000000;
    status = iree_string_builder_append_format(
        &output_string_builder,
        "{ \"invoke_time_ms\": %" PRId64 ", \"readback_time_ms\": %" PRId64
        ", \"outputs\": \"",
        invoke_time_ms, readback_time_ms);
  }
  if (iree_status_is_ok(status)) {
    status = print_outputs_from_call(call_state, &output_string_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&output_string_builder, "\"}");
  }

  if (iree_status_is_ok(status)) {
    // Note: this leaks the buffer. It's up to the caller to free it after use.
    char* outputs_string =
        strdup(iree_string_builder_buffer(&output_string_builder));
    iree_string_builder_deinitialize(&output_string_builder);
    call_state->callback_fn(outputs_string);
  } else {
    fprintf(stderr, "map_all_callback error:\n");
    // TODO(scotttodd): return a JSON object with the error message
    //   * free |status| and return a status to the loop with no storage
    //   * the returned string is always freed, so then we'd have no leaks
    iree_status_fprint(stderr, status);
    // Note: loop_emscripten.js must free 'status'!
    call_state->callback_fn(NULL);
  }

  iree_string_builder_deinitialize(&output_string_builder);
  iree_call_function_state_destroy(call_state);
  return status;
}

static iree_status_t allocate_mappable_device_buffer(
    iree_hal_device_t* device, iree_hal_buffer_view_t* buffer_view,
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;

  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  // Note: iree_hal_webgpu_simple_allocator_allocate_buffer only supports
  // CopySrc today, so we'll create the buffer directly with
  // wgpuDeviceCreateBuffer and then wrap it using iree_hal_webgpu_buffer_wrap.
  WGPUBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = "IREE_mapping",
      .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
      .size = data_length,
      .mappedAtCreation = false,
  };
  WGPUBuffer device_buffer_handle = NULL;
  // Note: wgpuBufferDestroy is called after iree_hal_webgpu_buffer_wrap ->
  //       iree_hal_buffer_release -> iree_hal_webgpu_buffer_destroy
  device_buffer_handle = wgpuDeviceCreateBuffer(
      iree_hal_webgpu_device_handle(device), &descriptor);
  if (!device_buffer_handle) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "unable to allocate buffer of size %" PRIdsz,
                            data_length);
  }
  const iree_hal_buffer_params_t target_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  return iree_hal_webgpu_buffer_wrap(
      device, iree_hal_device_allocator(device), target_params.type,
      target_params.access, target_params.usage, data_length,
      /*byte_offset=*/0,
      /*byte_length=*/data_length, device_buffer_handle,
      iree_allocator_system(), out_buffer);
}

// Processes outputs from a completed function invocation.
// Some output data types may require asynchronous mapping.
static iree_status_t process_call_outputs(
    iree_call_function_state_t* call_state) {
  call_state->readback_start_time = iree_time_now();

  iree_vm_list_t* outputs_list = iree_runtime_call_outputs(&call_state->call);
  iree_host_size_t outputs_size = iree_vm_list_size(outputs_list);
  iree_hal_device_t* device =
      iree_runtime_session_device(call_state->call.session);

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(iree_output_state_t) * outputs_size,
      (void**)&call_state->output_states));
  call_state->outputs_size = outputs_size;

  // TODO(scotttodd): allocate on the heap and track?
  //   * iree_loop_wait_all claims that wait_sources must live until the
  //     callback is issued
  //   * loop_emscripten uses what it needs (Promise handles/objects)
  //     immediately, before objects go out of scope
  iree_wait_source_t* wait_sources = (iree_wait_source_t*)iree_alloca(
      sizeof(iree_wait_source_t) * outputs_size);

  // Loop through the outputs once to find buffers that need readback.
  iree_host_size_t buffer_count = 0;
  for (iree_host_size_t i = 0; i < outputs_size; ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_variant_assign(outputs_list, i, &variant),
        "variant %" PRIhsz " not present", i);

    if (iree_vm_variant_is_ref(variant)) {
      if (!iree_hal_buffer_view_isa(variant.ref)) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "only buffer_view variants are supported");
      }

      // Output is a buffer_view ref, add to mapping batch (async).
      iree_event_initialize(false, &call_state->output_states[i].ready_event);
      buffer_count++;
      iree_hal_buffer_view_t* buffer_view =
          iree_hal_buffer_view_deref(variant.ref);
      call_state->output_states[i].buffer_view = buffer_view;
      iree_hal_buffer_view_retain(buffer_view);

      // TODO(scotttodd): signal event if failed
      IREE_RETURN_IF_ERROR(allocate_mappable_device_buffer(
          device, buffer_view,
          &call_state->output_states[i].mappable_device_buffer));
    } else {
      // Not a buffer, data is available immediately - start signaled.
      iree_event_initialize(true, &call_state->output_states[i].ready_event);
    }
    wait_sources[i] =
        iree_event_await(&call_state->output_states[i].ready_event);
  }

  // Loop through the outputs again to build a batched transfer command buffer.
  iree_hal_transfer_command_t* transfer_commands =
      (iree_hal_transfer_command_t*)iree_alloca(
          sizeof(iree_hal_transfer_command_t) * buffer_count);
  for (iree_host_size_t i = 0, buffer_index = 0; i < outputs_size; ++i) {
    iree_hal_buffer_view_t* buffer_view =
        call_state->output_states[i].buffer_view;
    if (!buffer_view) continue;

    iree_hal_buffer_t* source_buffer = iree_hal_buffer_view_buffer(buffer_view);
    iree_hal_buffer_t* source_buffer_allocated =
        iree_hal_buffer_allocated_buffer(source_buffer);
    iree_device_size_t source_offset =
        iree_hal_buffer_byte_offset(source_buffer);
    iree_hal_buffer_t* target_buffer =
        call_state->output_states[i].mappable_device_buffer;
    iree_device_size_t target_offset = 0;
    iree_device_size_t data_length =
        iree_hal_buffer_view_byte_length(buffer_view);

    transfer_commands[buffer_index++] = (iree_hal_transfer_command_t){
        .type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY,
        .copy =
            {
                .source_buffer = source_buffer_allocated,
                .source_offset = source_offset,
                .target_buffer = target_buffer,
                .target_offset = target_offset,
                .length = data_length,
            },
    };
  }

  // Construct and issue the transfer command buffer, then wait on it.
  iree_status_t status = iree_ok_status();
  iree_hal_command_buffer_t* transfer_command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_transfer_command_buffer(
        device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_QUEUE_AFFINITY_ANY, buffer_count, transfer_commands,
        &transfer_command_buffer);
  }
  iree_hal_semaphore_t* signal_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &signal_semaphore);
  }
  uint64_t signal_value = 1ull;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &signal_semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, 1, &transfer_command_buffer,
        /*binding_tables=*/NULL);
  }
  // TODO(scotttodd): Make this async - pass a wait source to iree_loop_wait_one
  //     1. create iree_hal_fence_t, iree_hal_fence_insert(fance, semaphore)
  //     2. iree_hal_fence_await -> iree_wait_source_t
  //     3. iree_loop_wait_one(loop, wait_source, ...)
  //   (requires moving off of nop_semaphore and wait source import)
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                     iree_infinite_timeout());
  }
  iree_hal_command_buffer_release(transfer_command_buffer);
  iree_hal_semaphore_release(signal_semaphore);

  // Loop through one last time to map the buffers asynchronously.
  for (iree_host_size_t i = 0; i < outputs_size; ++i) {
    if (!iree_status_is_ok(status)) break;
    if (!call_state->output_states[i].mappable_device_buffer) continue;

    iree_buffer_map_userdata_t* map_userdata = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), sizeof(*map_userdata), (void**)&map_userdata));

    map_userdata->call_state = call_state;
    map_userdata->buffer_index = i;
    iree_device_size_t data_length = iree_hal_buffer_view_byte_length(
        call_state->output_states[i].buffer_view);

    WGPUBuffer device_buffer = iree_hal_webgpu_buffer_handle(
        call_state->output_states[i].mappable_device_buffer);
    wgpuBufferMapAsync(device_buffer, WGPUMapMode_Read,
                       /*offset=*/0,
                       /*size=*/data_length, buffer_map_async_callback,
                       /*userdata=*/map_userdata);
  }

  // Finally, wait on all wait sources.
  //
  // If there are any buffer outputs that need asynchronous mapping, those
  // wait sources will be signaled when the mapping completes.
  //
  // Note: call_state (and everything within it) is kept alive until the
  // callback resolves.
  if (iree_status_is_ok(status)) {
    status = iree_loop_wait_all(iree_loop_emscripten(call_state->loop),
                                outputs_size, wait_sources,
                                iree_make_timeout_ms(5000), map_all_callback,
                                /*user_data=*/call_state);
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Function calling / invocations
//===----------------------------------------------------------------------===//

// Handles the completion callback from `iree_vm_async_invoke()`.
iree_status_t invoke_callback(void* user_data, iree_loop_t loop,
                              iree_status_t status, iree_vm_list_t* outputs) {
  iree_call_function_state_t* call_state =
      (iree_call_function_state_t*)user_data;
  call_state->invoke_end_time = iree_time_now();

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "iree_vm_async_invoke_callback_fn_t error:\n");
    iree_status_fprint(stderr, status);
    iree_call_function_state_destroy(call_state);
    return status;  // Note: loop_emscripten.js must free this!
  }

  status = process_call_outputs(call_state);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "process_call_outputs error:\n");
    iree_status_fprint(stderr, status);
    iree_call_function_state_destroy(call_state);
    // Note: loop_emscripten.js must free 'status'!
  }

  return status;
}

const bool call_function(
    iree_program_state_t* program_state, const char* function_name,
    const char* inputs,
    iree_call_function_callback_fn_t completion_callback_fn) {
  iree_status_t status = iree_ok_status();

  iree_call_function_state_t* call_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(), sizeof(*call_state),
                                   (void**)&call_state);
  }
  call_state->loop = program_state->sample_state->loop;
  call_state->callback_fn = completion_callback_fn;

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
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        program_state->session, iree_string_builder_view(&name_builder),
        &call_state->call);
  }
  iree_string_builder_deinitialize(&name_builder);

  if (iree_status_is_ok(status)) {
    status = parse_inputs_into_call(
        &call_state->call, iree_runtime_session_device(program_state->session),
        iree_runtime_session_device_allocator(program_state->session),
        iree_make_cstring_view(inputs));
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(),
                                   sizeof(*(call_state->invoke_state)),
                                   (void**)&(call_state->invoke_state));
  }

  if (iree_status_is_ok(status)) {
    iree_loop_t loop = iree_loop_emscripten(program_state->sample_state->loop);
    iree_vm_context_t* vm_context =
        iree_runtime_session_context(call_state->call.session);
    iree_vm_function_t vm_function = call_state->call.function;
    iree_vm_list_t* inputs = call_state->call.inputs;
    iree_vm_list_t* outputs = call_state->call.outputs;

    call_state->invoke_start_time = iree_time_now();
    status = iree_vm_async_invoke(
        loop, call_state->invoke_state, vm_context, vm_function,
        IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL, inputs, outputs,
        iree_allocator_system(), invoke_callback, /*user_data=*/call_state);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_call_function_state_destroy(call_state);
    return false;
  }

  return true;
}
