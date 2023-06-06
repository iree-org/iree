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
//   For example, the CLI `--function_input=f32=1 --function_input=f32=2`
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

extern iree_status_t create_device(iree_allocator_t host_allocator,
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
    status = create_device(iree_allocator_system(), &sample_state->device);
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

typedef struct iree_buffer_map_userdata_t {
  iree_hal_buffer_view_t* source_buffer_view;
  iree_hal_buffer_t* readback_buffer;
} iree_buffer_map_userdata_t;

static void iree_webgpu_mapped_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  WGPUBuffer buffer_handle = (WGPUBuffer)user_data;
  wgpuBufferUnmap(buffer_handle);
}

// TODO(scotttodd): move async mapping into webgpu/buffer.h/.c?
static void buffer_map_sync_callback(WGPUBufferMapAsyncStatus map_status,
                                     void* userdata_ptr) {
  iree_buffer_map_userdata_t* userdata =
      (iree_buffer_map_userdata_t*)userdata_ptr;
  switch (map_status) {
    case WGPUBufferMapAsyncStatus_Success:
      break;
    case WGPUBufferMapAsyncStatus_Error:
      fprintf(stderr, "  buffer_map_sync_callback status: Error\n");
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      fprintf(stderr, "  buffer_map_sync_callback status: DeviceLost\n");
      break;
    case WGPUBufferMapAsyncStatus_Unknown:
    default:
      fprintf(stderr, "  buffer_map_sync_callback status: Unknown\n");
      break;
  }

  if (map_status != WGPUBufferMapAsyncStatus_Success) {
    iree_hal_buffer_view_release(userdata->source_buffer_view);
    iree_hal_buffer_release(userdata->readback_buffer);
    iree_allocator_free(iree_allocator_system(), userdata);
    return;
  }

  iree_status_t status = iree_ok_status();

  // TODO(scotttodd): bubble result(s) up to the caller (async + callback API)

  iree_device_size_t data_offset = iree_hal_buffer_byte_offset(
      iree_hal_buffer_view_buffer(userdata->source_buffer_view));
  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(userdata->source_buffer_view);
  WGPUBuffer buffer_handle =
      iree_hal_webgpu_buffer_handle(userdata->readback_buffer);

  // For this sample we want to print arbitrary buffers, which is easiest
  // using the |iree_hal_buffer_view_format| function. Internally, that
  // function requires synchronous buffer mapping, so we'll first wrap the
  // already (async) mapped GPU memory into a heap buffer. In a less general
  // application (or one not requiring pretty logging like this), we could
  // skip a few buffer copies and other data transformations here.

  const void* data_ptr =
      wgpuBufferGetConstMappedRange(buffer_handle, data_offset, data_length);

  iree_hal_buffer_t* heap_buffer = NULL;
  if (iree_status_is_ok(status)) {
    // The buffer we get from WebGPU may not be aligned to 64.
    iree_hal_memory_access_t memory_access =
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_UNALIGNED;
    status = iree_hal_heap_buffer_wrap(
        userdata->readback_buffer->device_allocator,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL, memory_access,
        IREE_HAL_BUFFER_USAGE_MAPPING, data_length,
        iree_make_byte_span((void*)data_ptr, data_length),
        (iree_hal_buffer_release_callback_t){
            .fn = iree_webgpu_mapped_buffer_release,
            .user_data = buffer_handle,
        },
        &heap_buffer);
  }

  // Copy the original buffer_view, backed by the mapped heap buffer instead.
  iree_hal_buffer_view_t* heap_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_create_like(
        heap_buffer, userdata->source_buffer_view, iree_allocator_system(),
        &heap_buffer_view);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Call output:\n");
    status = iree_hal_buffer_view_fprint(stdout, heap_buffer_view,
                                         /*max_element_count=*/4096,
                                         iree_allocator_system());
    fprintf(stdout, "\n");
  }
  iree_hal_buffer_view_release(heap_buffer_view);
  iree_hal_buffer_release(heap_buffer);

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "buffer_map_sync_callback error:\n");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  iree_hal_buffer_view_release(userdata->source_buffer_view);
  iree_hal_buffer_release(userdata->readback_buffer);
  iree_allocator_free(iree_allocator_system(), userdata);
}

static iree_status_t print_buffer_view(iree_hal_device_t* device,
                                       iree_hal_buffer_view_t* buffer_view) {
  iree_status_t status = iree_ok_status();

  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_device_size_t data_offset = iree_hal_buffer_byte_offset(buffer);
  iree_device_size_t data_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  // ----------------------------------------------
  // Allocate mappable host memory.
  // Note: iree_hal_webgpu_simple_allocator_allocate_buffer only supports
  // CopySrc today, so we'll create the buffer directly with
  // wgpuDeviceCreateBuffer and then wrap it using iree_hal_webgpu_buffer_wrap.
  WGPUBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = "IREE_readback",
      .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
      .size = data_length,
      .mappedAtCreation = false,
  };
  WGPUBuffer readback_buffer_handle = NULL;
  if (iree_status_is_ok(status)) {
    readback_buffer_handle = wgpuDeviceCreateBuffer(
        iree_hal_webgpu_device_handle(device), &descriptor);
    if (!readback_buffer_handle) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "unable to allocate buffer of size %" PRIdsz,
                                data_length);
    }
  }
  iree_device_size_t target_offset = 0;
  const iree_hal_buffer_params_t target_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  iree_hal_buffer_t* readback_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_buffer_wrap(
        device, iree_hal_device_allocator(device), target_params.type,
        target_params.access, target_params.usage, data_length,
        /*byte_offset=*/0,
        /*byte_length=*/data_length, readback_buffer_handle,
        iree_allocator_system(), &readback_buffer);
  }
  // ----------------------------------------------

  // ----------------------------------------------
  // Transfer from device memory to mappable host memory.
  const iree_hal_transfer_command_t transfer_command = {
      .type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY,
      .copy =
          {
              .source_buffer = buffer,
              .source_offset = data_offset,
              .target_buffer = readback_buffer,
              .target_offset = target_offset,
              .length = data_length,
          },
  };
  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_transfer_command_buffer(
        device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_QUEUE_AFFINITY_ANY, /*transfer_count=*/1, &transfer_command,
        &command_buffer);
  }
  iree_hal_semaphore_t* fence_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, 0ull, &fence_semaphore);
  }
  uint64_t signal_value = 1ull;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &fence_semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, 1, &command_buffer);
  }
  // TODO(scotttodd): Make this async - pass a wait source to iree_loop_wait_one
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(fence_semaphore, signal_value,
                                     iree_infinite_timeout());
  }
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(fence_semaphore);
  // ----------------------------------------------

  iree_buffer_map_userdata_t* userdata = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(),
                                   sizeof(iree_buffer_map_userdata_t),
                                   (void**)&userdata);
    iree_hal_buffer_view_retain(buffer_view);  // Released in the callback.
    userdata->source_buffer_view = buffer_view;
    userdata->readback_buffer = readback_buffer;
  }

  if (iree_status_is_ok(status)) {
    wgpuBufferMapAsync(readback_buffer_handle, WGPUMapMode_Read, /*offset=*/0,
                       /*size=*/data_length, buffer_map_sync_callback,
                       /*userdata=*/userdata);
  }

  return status;
}

static iree_status_t print_outputs_from_call(
    iree_runtime_call_t* call, iree_string_builder_t* outputs_builder) {
  iree_vm_list_t* variants_list = iree_runtime_call_outputs(call);
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
      if (iree_hal_buffer_view_isa(variant.ref)) {
        iree_hal_buffer_view_t* buffer_view =
            iree_hal_buffer_view_deref(variant.ref);
        // TODO(scotttodd): join async outputs together and return to caller
        iree_hal_device_t* device = iree_runtime_session_device(call->session);
        IREE_RETURN_IF_ERROR(print_buffer_view(device, buffer_view));
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

iree_status_t invoke_callback(void* user_data, iree_loop_t loop,
                              iree_status_t status, iree_vm_list_t* outputs) {
  iree_vm_async_invoke_state_t* invoke_state =
      (iree_vm_async_invoke_state_t*)user_data;

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "iree_vm_async_invoke_callback_fn_t error:\n");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  iree_vm_list_release(outputs);

  iree_allocator_free(iree_allocator_system(), (void*)invoke_state);
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

  // TODO(scotttodd): benchmark iterations (somehow with async)

  iree_vm_async_invoke_state_t* invoke_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(),
                                   sizeof(iree_vm_async_invoke_state_t),
                                   (void**)&invoke_state);
  }
  // TODO(scotttodd): emscripten / browser loop here
  iree_status_t loop_status = iree_ok_status();
  iree_loop_t loop = iree_loop_inline(&loop_status);
  if (iree_status_is_ok(status)) {
    iree_vm_context_t* vm_context = iree_runtime_session_context(call.session);
    iree_vm_function_t vm_function = call.function;
    iree_vm_list_t* inputs = call.inputs;
    iree_vm_list_t* outputs = call.outputs;

    status = iree_vm_async_invoke(loop, invoke_state, vm_context, vm_function,
                                  IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
                                  inputs, outputs, iree_allocator_system(),
                                  invoke_callback,
                                  /*user_data=*/invoke_state);
  }

  // TODO(scotttodd): record end time in async callback instead of here
  // TODO(scotttodd): print outputs in async callback instead of here

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
  char* outputs_string = strdup(iree_string_builder_buffer(&outputs_builder));
  iree_string_builder_deinitialize(&outputs_builder);
  return outputs_string;
}
