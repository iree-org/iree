// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/module.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

// Limit the number of bindings we pass down through the HAL. This can be tuned
// in the future but right now guards the stack from blowing up during calls.
#define IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT ((iree_host_size_t)32)

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_hal_allocator_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_view_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_command_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_layout_descriptor =
    {0};
static iree_vm_ref_type_descriptor_t iree_hal_device_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_event_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_layout_descriptor = {
    0};
static iree_vm_ref_type_descriptor_t iree_hal_semaphore_descriptor = {0};

#define IREE_VM_REGISTER_HAL_C_TYPE(type, name, destroy_fn, descriptor)   \
  descriptor.type_name = iree_make_cstring_view(name);                    \
  descriptor.offsetof_counter = offsetof(iree_hal_resource_t, ref_count); \
  descriptor.destroy = (iree_vm_ref_destroy_t)destroy_fn;                 \
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&descriptor));

IREE_API_EXPORT iree_status_t iree_hal_module_register_types(void) {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_allocator_t, "hal.allocator",
                              iree_hal_allocator_destroy,
                              iree_hal_allocator_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_t, "hal.buffer",
                              iree_hal_buffer_recycle,
                              iree_hal_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_view_t, "hal.buffer_view",
                              iree_hal_buffer_view_destroy,
                              iree_hal_buffer_view_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_command_buffer_t, "hal.command_buffer",
                              iree_hal_command_buffer_destroy,
                              iree_hal_command_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_t, "hal.descriptor_set",
                              iree_hal_descriptor_set_destroy,
                              iree_hal_descriptor_set_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_layout_t,
                              "hal.descriptor_set_layout",
                              iree_hal_descriptor_set_layout_destroy,
                              iree_hal_descriptor_set_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_device_t, "hal.device",
                              iree_hal_device_destroy,
                              iree_hal_device_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_event_t, "hal.event",
                              iree_hal_event_destroy,
                              iree_hal_event_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_t, "hal.executable",
                              iree_hal_executable_destroy,
                              iree_hal_executable_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_layout_t,
                              "hal.executable_layout",
                              iree_hal_executable_layout_destroy,
                              iree_hal_executable_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_semaphore_t, "hal.semaphore",
                              iree_hal_semaphore_destroy,
                              iree_hal_semaphore_descriptor);

  has_registered = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_buffer_view_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_command_buffer,
                             iree_hal_command_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set,
                             iree_hal_descriptor_set_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                             iree_hal_descriptor_set_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_event, iree_hal_event_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_layout,
                             iree_hal_executable_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_semaphore, iree_hal_semaphore_t);

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_hal_module_t {
  iree_allocator_t host_allocator;
  iree_hal_device_t* shared_device;
  // TODO(benvanik): types.
} iree_hal_module_t;

#define IREE_HAL_MODULE_CAST(module) \
  (iree_hal_module_t*)((uint8_t*)(module) + iree_vm_native_module_size());

typedef struct iree_hal_module_state_t {
  iree_allocator_t host_allocator;
  iree_hal_device_t* shared_device;
  iree_status_t loop_status;
  iree_hal_executable_cache_t* executable_cache;

  iree_hal_semaphore_t* submit_semaphore;
  uint64_t submit_value;
} iree_hal_module_state_t;

static void IREE_API_PTR iree_hal_module_destroy(void* base_module) {
  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);
  iree_hal_device_release(module->shared_device);
}

static iree_status_t IREE_API_PTR
iree_hal_module_alloc_state(void* self, iree_allocator_t host_allocator,
                            iree_vm_module_state_t** out_module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(self);
  iree_hal_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->shared_device = module->shared_device;
  iree_hal_device_retain(state->shared_device);

  state->loop_status = iree_ok_status();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_cache_create(
              state->shared_device, iree_string_view_empty(),
              iree_loop_inline(&state->loop_status), &state->executable_cache));

  state->submit_value = 0ull;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_create(state->shared_device, state->submit_value,
                                    &state->submit_semaphore));

  *out_module_state = (iree_vm_module_state_t*)state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR
iree_hal_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  iree_hal_semaphore_release(state->submit_semaphore);
  iree_hal_executable_cache_release(state->executable_cache);
  iree_status_ignore(state->loop_status);
  iree_hal_device_release(state->shared_device);
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_hal_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  switch (signal) {
    case IREE_VM_SIGNAL_SUSPEND:
    case IREE_VM_SIGNAL_LOW_MEMORY:
      return iree_hal_device_trim(state->shared_device);
    default:
      return iree_ok_status();
  }
}

//===----------------------------------------------------------------------===//
// Experimental APIs
//===----------------------------------------------------------------------===//
// NOTE: Ex* APIs are experimental and likely to be removed soon. Modules
// using these APIs are not forward compatible.

IREE_VM_ABI_EXPORT(iree_hal_module_ex_shared_device,  //
                   iree_hal_module_state_t,           //
                   v, r) {
  rets->r0 = iree_hal_device_retain_ref(state->shared_device);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_ex_submit_and_wait,  //
                   iree_hal_module_state_t,             //
                   rr, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r1, &command_buffer));

  // Batch with our single command buffer.
  iree_hal_submission_batch_t batch;
  memset(&batch, 0, sizeof(batch));

  iree_hal_command_buffer_t* command_buffer_ptrs[] = {command_buffer};
  batch.command_buffer_count = IREE_ARRAYSIZE(command_buffer_ptrs);
  batch.command_buffers = command_buffer_ptrs;

  uint64_t next_semaphore_value = ++state->submit_value;
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {state->submit_semaphore};
  uint64_t signal_semaphore_values[] = {next_semaphore_value};
  batch.signal_semaphores.count = IREE_ARRAYSIZE(signal_semaphore_ptrs);
  batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
  batch.signal_semaphores.payload_values = signal_semaphore_values;

  iree_status_t status = iree_hal_device_submit_and_wait(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, 0, 1, &batch,
      state->submit_semaphore, next_semaphore_value, iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    return status;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_allocate,  //
                   iree_hal_module_state_t,             //
                   riii, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i1;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i2;
  iree_vm_size_t allocation_size = (iree_vm_size_t)args->i3;

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, iree_const_byte_span_empty(),
      &buffer));
  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

static void iree_hal_module_mapped_buffer_release(void* user_data,
                                                  iree_hal_buffer_t* buffer) {
  iree_vm_buffer_t* backing_buffer = (iree_vm_buffer_t*)user_data;
  iree_vm_buffer_release(backing_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_map_byte_buffer,  //
                   iree_hal_module_state_t,                    //
                   riiirii, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  bool is_try = args->i1 != 0;
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i2;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i3;
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r4, &source));
  iree_vm_size_t offset = (iree_vm_size_t)args->i5;
  iree_vm_size_t length = (iree_vm_size_t)args->i6;

  iree_host_size_t buffer_length = source->data.data_length;
  if (length == -1) {
    length = buffer_length;
  }
  if (length < 0 || offset < 0 || offset > buffer_length ||
      offset + length > buffer_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "byte range out of bounds (requested %d-%d of available %zu)", offset,
        (offset + length - 1), buffer_length);
  }

  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_READ;
  if (!iree_all_bits_set(source->access, IREE_VM_BUFFER_ACCESS_MUTABLE)) {
    // Source buffer is read-only; require that the access request matches.
    if (!iree_all_bits_set(buffer_usage, IREE_HAL_BUFFER_USAGE_CONSTANT)) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "source buffer is immutable and can only be "
                              "mapped for constant usage");
    }

    // NOTE: if we wanted to lock things down for when there's no MMU to ensure
    // that the loaded program doesn't touch the memory then we could just fail
    // the request - the program will then perform an alloc+copy and can do
    // whatever it wants with the memory.
  } else {
    // Source buffer is mutable; allow in-place writes.
    if (!iree_all_bits_set(buffer_usage, IREE_HAL_BUFFER_USAGE_CONSTANT)) {
      allowed_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
    }
  }

  // Try mapping - note that this may fail if the target device cannot map the
  // memory into the given type (for example, mapping a host buffer into
  // device-local memory is only going to work on unified memory systems).
  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
      .access = allowed_access,
  };
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = length,
      .handle.host_allocation.ptr = source->data.data + offset,
  };
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_module_mapped_buffer_release,
      .user_data = source,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer, release_callback, &buffer);
  if (iree_status_is_ok(status)) {
    // Mapping succeeded - retain the source buffer that'll be released by
    // iree_hal_module_map_data_ctl when the mapping is no longer used.
    iree_vm_buffer_retain(source);
    rets->r0 = iree_hal_buffer_move_ref(buffer);
    return iree_ok_status();
  }

  // Failed to map - if this was a try then don't fail and just rely on the
  // result being nullptr to indicate to the caller that things failed.
  memset(&rets->r0, 0, sizeof(rets->r0));
  if (is_try) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}

// TODO(#7277): drop this method (use map instead) with streams.
IREE_VM_ABI_EXPORT(iree_hal_module_allocator_wrap_byte_buffer,  //
                   iree_hal_module_state_t,                     //
                   riirii, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i1;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i2;
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r3, &source));
  iree_vm_size_t offset = (iree_vm_size_t)args->i4;
  iree_vm_size_t length = (iree_vm_size_t)args->i5;

  iree_host_size_t buffer_length = source->data.data_length;
  if (length == -1) {
    length = buffer_length;
  }
  if (length < 0 || offset < 0 || offset > buffer_length ||
      offset + length > buffer_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "byte range out of bounds (requested %d-%d of available %zu)", offset,
        (offset + length - 1), buffer_length);
  }

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(
          allocator, params, length,
          iree_make_const_byte_span(source->data.data + offset, length),
          &buffer),
      "failed to allocate buffer of length %d", length);

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_assert,  //
                   iree_hal_module_state_t,        //
                   rrriii, v) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  iree_vm_buffer_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &message));
  iree_string_view_t message_str IREE_ATTRIBUTE_UNUSED =
      iree_vm_buffer_as_string(message);
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r2, &allocator));
  iree_vm_size_t minimum_length = (iree_vm_size_t)args->i3;
  iree_hal_memory_type_t required_memory_types =
      (iree_hal_memory_type_t)args->i4;
  iree_hal_buffer_usage_t required_buffer_usage =
      (iree_hal_buffer_usage_t)args->i5;

  // Ensure we have enough bytes in the buffer for the encoding we have.
  // Note that having more bytes is fine:
  //   assert(expected_length <= actual_length);
  iree_device_size_t actual_length = iree_hal_buffer_byte_length(buffer);
  if (actual_length < minimum_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s buffer byte length %" PRIdsz " less than expected minimum %d",
        (int)message_str.size, message_str.data, actual_length, minimum_length);
  }

  // TODO(benvanik): assert that the buffer view is accessible from the
  // target device. This needs some iree_hal_allocator_* methods for checking
  // whether the external buffer can be used. To start we just compare if the
  // allocators are identical.

  // All memory type bits expected (indicating where the program intends to use
  // the buffer data) must be set in the buffer while the buffer is allowed to
  // have more bits.
  iree_hal_memory_type_t actual_memory_type =
      iree_hal_buffer_memory_type(buffer);
  if (!iree_all_bits_set(actual_memory_type, required_memory_types)) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t actual_memory_type_str =
        iree_hal_memory_type_format(actual_memory_type, &temp0);
    iree_string_view_t expected_memory_type_str =
        iree_hal_memory_type_format(required_memory_types, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s buffer memory type is not compatible; buffer has %.*s, operation "
        "requires %.*s",
        (int)message_str.size, message_str.data,
        (int)actual_memory_type_str.size, actual_memory_type_str.data,
        (int)expected_memory_type_str.size, expected_memory_type_str.data);
#else
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s buffer memory type is not compatible; buffer has %08X, operation "
        "requires %08X",
        (int)message_str.size, message_str.data, actual_memory_type,
        expected_memory_type);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  // All usage bits expected (indicating what the program intends to use the
  // buffer for) must be set in the buffer while the buffer is allowed to have
  // more bits.
  iree_hal_buffer_usage_t actual_buffer_usage =
      iree_hal_buffer_allowed_usage(buffer);
  if (!iree_all_bits_set(actual_buffer_usage, required_buffer_usage)) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str =
        iree_hal_buffer_usage_format(actual_buffer_usage, &temp0);
    iree_string_view_t required_usage_str =
        iree_hal_buffer_usage_format(required_buffer_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s requested usage was not specified when the buffer was allocated; "
        "buffer allows %.*s, operation requires %.*s",
        (int)message_str.size, message_str.data, (int)allowed_usage_str.size,
        allowed_usage_str.data, (int)required_usage_str.size,
        required_usage_str.data);
#else
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s requested usage was not specified when the buffer was allocated; "
        "buffer allows %08X, operation requires %08X",
        (int)message_str.size, message_str.data, allowed_buffer_usage,
        required_buffer_usage);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_subspan,  //
                   iree_hal_module_state_t,         //
                   rii, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_vm_size_t source_offset = (iree_vm_size_t)args->i1;
  iree_vm_size_t length = (iree_vm_size_t)args->i2;

  iree_hal_buffer_t* subspan_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_subspan(source_buffer, source_offset, length,
                              &subspan_buffer),
      "invalid subspan of an existing buffer (source_offset=%d, length=%d)",
      source_offset, length);
  rets->r0 = iree_hal_buffer_move_ref(subspan_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_length,  //
                   iree_hal_module_state_t,        //
                   r, i) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->i0 = iree_hal_buffer_byte_length(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_load,  //
                   iree_hal_module_state_t,      //
                   rii, i) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_vm_size_t source_offset = (iree_vm_size_t)args->i1;
  iree_vm_size_t length = (iree_vm_size_t)args->i2;

  uint32_t target_buffer = 0;
  if (length > sizeof(target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "load length byte count %d exceeds max", length);
  }

  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      state->shared_device, source_buffer, source_offset, &target_buffer,
      length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  rets->i0 = target_buffer;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_store,  //
                   iree_hal_module_state_t,       //
                   irii, v) {
  int32_t value = args->i0;
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &target_buffer));
  iree_vm_size_t target_offset = (iree_vm_size_t)args->i2;
  iree_vm_size_t length = (iree_vm_size_t)args->i3;

  if (length > sizeof(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "store length byte count %d exceeds max", length);
  } else if (target_offset + length >
             iree_hal_buffer_byte_length(target_buffer)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "store out of bounds (target_offset=%d, length=%d into max %" PRIdsz
        ")",
        target_offset, length, iree_hal_buffer_byte_length(target_buffer));
  }

  return iree_hal_device_transfer_h2d(
      state->shared_device, &value, target_buffer, target_offset, length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_create,  //
                   iree_hal_module_state_t,             //
                   riiCiD, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_hal_element_type_t element_type = (iree_hal_element_type_t)args->i1;
  iree_hal_encoding_type_t encoding_type = (iree_hal_encoding_type_t)args->i2;
  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t* shape_dims = NULL;
  IREE_VM_ABI_VLA_STACK_CAST(args, a3_count, a3, iree_hal_dim_t, 128,
                             &shape_rank, &shape_dims);

  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      source_buffer, shape_dims, shape_rank, element_type, encoding_type,
      state->host_allocator, &buffer_view));
  rets->r0 = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_ok_status();
}

// Returns true if the |expected_type| can be satisfied with |actual_type|.
// This allows for basic type widening and bypassing instead of requiring an
// exact match in all cases.
static bool iree_hal_element_types_are_compatible(
    iree_hal_element_type_t actual_type,
    iree_hal_element_type_t expected_type) {
  if (iree_hal_element_numerical_type_is_opaque(actual_type)) {
    // If the provided type is opaque it can map to anything. This allows
    // applications to bypass the checks when they are treating all the data as
    // opaque, such as when carrying around buffer data in binary blobs.
    return true;
  }

  if (iree_hal_element_numerical_type_is_integer(actual_type) &&
      iree_hal_element_numerical_type_is_integer(expected_type) &&
      iree_hal_element_bit_count(actual_type) ==
          iree_hal_element_bit_count(expected_type)) {
    // Integer types of the same bit width are allowed to be cast.
    // This allows users or the compiler to treat data as signless while still
    // allowing signedness. For example, tensor<1xi32> can successfully match
    // a tensor<1xui32> expectation.
    return true;
  }

  // Otherwise we require an exact match. This may be overly conservative but
  // in most cases is a useful error message. Users can pass in OPAQUE types if
  // hitting this to bypass.
  return actual_type == expected_type;
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_assert,  //
                   iree_hal_module_state_t,             //
                   rriiCiD, v) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  iree_vm_buffer_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &message));
  iree_string_view_t message_str IREE_ATTRIBUTE_UNUSED =
      iree_vm_buffer_as_string(message);
  iree_hal_element_type_t expected_element_type =
      (iree_hal_element_type_t)args->i2;
  iree_hal_encoding_type_t expected_encoding_type =
      (iree_hal_encoding_type_t)args->i3;
  iree_host_size_t expected_shape_rank = 0;
  iree_hal_dim_t* expected_shape_dims = NULL;
  IREE_VM_ABI_VLA_STACK_CAST(args, a4_count, a4, iree_hal_dim_t, 128,
                             &expected_shape_rank, &expected_shape_dims);

  // Check encoding first; getting the encoding wrong is worse than the shape.
  // If the actual encoding is opaque we allow it to pass through - this lets
  // users override the assertion in the case where they are just passing data
  // around and don't care about the contents.
  iree_hal_encoding_type_t actual_encoding_type =
      iree_hal_buffer_view_encoding_type(buffer_view);
  if (actual_encoding_type != IREE_HAL_ENCODING_TYPE_OPAQUE &&
      actual_encoding_type != expected_encoding_type) {
    // TODO(benvanik): string formatting of encodings.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s encoding mismatch; expected %08X but have %08X",
        (int)message_str.size, message_str.data, expected_encoding_type,
        actual_encoding_type);
  }

  // Element types determine the storage requirements.
  // If the actual element type is opaque we allow it to pass through.
  iree_hal_element_type_t actual_element_type =
      iree_hal_buffer_view_element_type(buffer_view);
  if (!iree_hal_element_types_are_compatible(actual_element_type,
                                             expected_element_type)) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
    char actual_element_type_str[32];
    iree_host_size_t actual_element_type_str_length = 0;
    char expected_element_type_str[32];
    iree_host_size_t expected_element_type_str_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
        actual_element_type, sizeof(actual_element_type_str),
        actual_element_type_str, &actual_element_type_str_length));
    IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
        expected_element_type, sizeof(expected_element_type_str),
        expected_element_type_str, &expected_element_type_str_length));
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s element type mismatch; expected %.*s (%08X) but have %.*s (%08X)",
        (int)message_str.size, message_str.data,
        (int)expected_element_type_str_length, expected_element_type_str,
        expected_element_type, (int)actual_element_type_str_length,
        actual_element_type_str, actual_element_type);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s element type mismatch; expected %08X but have %08X",
        (int)message_str.size, message_str.data, expected_element_type,
        actual_element_type);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  // Rank check before the individual shape dimensions.
  iree_host_size_t actual_shape_rank =
      iree_hal_buffer_view_shape_rank(buffer_view);
  const iree_hal_dim_t* actual_shape_dims =
      iree_hal_buffer_view_shape_dims(buffer_view);
  iree_status_t shape_status = iree_ok_status();
  if (actual_shape_rank != expected_shape_rank) {
    shape_status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "%.*s shape rank mismatch; expected %zu but have %zu",
                         (int)message_str.size, message_str.data,
                         expected_shape_rank, actual_shape_rank);
  }
  if (iree_status_is_ok(shape_status)) {
    for (iree_host_size_t i = 0; i < actual_shape_rank; ++i) {
      if (actual_shape_dims[i] == expected_shape_dims[i]) continue;
      // Dimension mismatch.
      shape_status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%.*s shape dimension %zu mismatch; expected %d but have %d",
          (int)message_str.size, message_str.data, i, expected_shape_dims[i],
          actual_shape_dims[i]);
      break;
    }
  }

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
  if (!iree_status_is_ok(shape_status)) {
    char actual_shape_str[32];
    iree_host_size_t actual_shape_str_length = 0;
    char expected_shape_str[32];
    iree_host_size_t expected_shape_str_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_format_shape(
        actual_shape_dims, actual_shape_rank, sizeof(actual_shape_str),
        actual_shape_str, &actual_shape_str_length));
    IREE_RETURN_IF_ERROR(iree_hal_format_shape(
        expected_shape_dims, expected_shape_rank, sizeof(expected_shape_str),
        expected_shape_str, &expected_shape_str_length));
    shape_status = iree_status_annotate_f(
        shape_status, "expected shape %.*s, actual shape %.*s",
        (int)expected_shape_str_length, expected_shape_str,
        (int)actual_shape_str_length, actual_shape_str);
  }
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

  return shape_status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_buffer,  //
                   iree_hal_module_state_t,             //
                   r, r) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->r0 =
      iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_byte_length,  //
                   iree_hal_module_state_t,                  //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (iree_vm_size_t)iree_hal_buffer_view_byte_length(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_element_type,  //
                   iree_hal_module_state_t,                   //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (uint32_t)iree_hal_buffer_view_element_type(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_encoding_type,  //
                   iree_hal_module_state_t,                    //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (uint32_t)iree_hal_buffer_view_encoding_type(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_rank,  //
                   iree_hal_module_state_t,           //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (iree_vm_size_t)iree_hal_buffer_view_shape_rank(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_dim,  //
                   iree_hal_module_state_t,          //
                   ri, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  iree_vm_size_t index = (iree_vm_size_t)args->i1;
  rets->i0 = (iree_vm_size_t)iree_hal_buffer_view_shape_dim(buffer_view, index);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_trace,  //
                   iree_hal_module_state_t,            //
                   rCrD, v) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE

  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r0, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  fprintf(stderr, "=== %.*s ===\n", (int)key_str.size, key_str.data);
  for (iree_host_size_t i = 0; i < args->a1_count; ++i) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_check_deref(args->a1[i].r0, &buffer_view));

    // NOTE: this export is for debugging only and a no-op in min-size builds.
    // We heap-alloc here because at the point this export is used performance
    // is not a concern.

    // Query total length (excluding NUL terminator).
    iree_host_size_t result_length = 0;
    iree_status_t status = iree_hal_buffer_view_format(buffer_view, SIZE_MAX, 0,
                                                       NULL, &result_length);
    if (!iree_status_is_out_of_range(status)) {
      return status;
    }
    ++result_length;  // include NUL

    // Allocate scratch heap memory to contain the result and format into it.
    char* result_str = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        state->host_allocator, result_length, (void**)&result_str));
    status = iree_hal_buffer_view_format(buffer_view, SIZE_MAX, result_length,
                                         result_str, &result_length);
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "%.*s\n", (int)result_length, result_str);
    }
    iree_allocator_free(state->host_allocator, result_str);
    IREE_RETURN_IF_ERROR(status);
  }
  fprintf(stderr, "\n");

#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_create,  //
                   iree_hal_module_state_t,                //
                   rii, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_command_buffer_mode_t modes =
      (iree_hal_command_buffer_mode_t)args->i1;
  iree_hal_command_category_t command_categories =
      (iree_hal_command_category_t)args->i2;

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device, modes, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));
  rets->r0 = iree_hal_command_buffer_move_ref(command_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_begin,  //
                   iree_hal_module_state_t,               //
                   r, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));

  return iree_hal_command_buffer_begin(command_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_end,  //
                   iree_hal_module_state_t,             //
                   r, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));

  return iree_hal_command_buffer_end(command_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_begin_debug_group,  //
                   iree_hal_module_state_t,                           //
                   rr, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_vm_buffer_t* label = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &label));
  iree_string_view_t label_str = iree_vm_buffer_as_string(label);
  // TODO(benvanik): query from VM.
  iree_hal_label_location_t location = {
      .file = iree_string_view_empty(),
      .line = 0,
  };
  iree_hal_command_buffer_begin_debug_group(
      command_buffer, label_str, iree_hal_label_color_unspecified(), &location);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_end_debug_group,  //
                   iree_hal_module_state_t,                         //
                   r, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_command_buffer_end_debug_group(command_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_execution_barrier,  //
                   iree_hal_module_state_t,                           //
                   riii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_execution_stage_t source_stage_mask =
      (iree_hal_execution_stage_t)args->i1;
  iree_hal_execution_stage_t target_stage_mask =
      (iree_hal_execution_stage_t)args->i2;
  iree_hal_execution_barrier_flags_t flags =
      (iree_hal_execution_barrier_flags_t)args->i3;

  // TODO(benvanik): decode barriers.
  iree_hal_memory_barrier_t global_barrier;
  global_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  global_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;

  return iree_hal_command_buffer_execution_barrier(
      command_buffer, source_stage_mask, target_stage_mask, flags, 1,
      &global_barrier, 0, NULL);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_fill_buffer,  //
                   iree_hal_module_state_t,                     //
                   rriiii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &target_buffer));
  iree_vm_size_t target_offset = (iree_vm_size_t)args->i2;
  iree_vm_size_t length = (iree_vm_size_t)args->i3;
  uint32_t pattern = (uint32_t)args->i4;
  uint32_t pattern_length = (uint32_t)args->i5;
  return iree_hal_command_buffer_fill_buffer(command_buffer, target_buffer,
                                             target_offset, length, &pattern,
                                             pattern_length);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_copy_buffer,  //
                   iree_hal_module_state_t,                     //
                   rririi, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &source_buffer));
  iree_vm_size_t source_offset = (iree_vm_size_t)args->i2;
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r3, &target_buffer));
  iree_vm_size_t target_offset = (iree_vm_size_t)args->i4;
  iree_vm_size_t length = (iree_vm_size_t)args->i5;
  return iree_hal_command_buffer_copy_buffer(command_buffer, source_buffer,
                                             source_offset, target_buffer,
                                             target_offset, length);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_push_constants,  //
                   iree_hal_module_state_t,                        //
                   rriCiD, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_layout_check_deref(args->r1, &executable_layout));
  iree_vm_size_t offset = (iree_vm_size_t)args->i2;
  iree_host_size_t value_count = args->a3_count;
  const uint32_t* values = (const uint32_t*)&args->a3[0].i0;

  return iree_hal_command_buffer_push_constants(
      command_buffer, executable_layout, offset * sizeof(uint32_t), values,
      value_count * sizeof(uint32_t));
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_push_descriptor_set,  //
                   iree_hal_module_state_t,                             //
                   rriCiriiD, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_layout_check_deref(args->r1, &executable_layout));
  iree_vm_size_t set = args->i2;

  iree_host_size_t binding_count = args->a3_count;
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "binding count %zu > %zu",
                            binding_count,
                            IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_descriptor_set_binding_t* bindings =
      (iree_hal_descriptor_set_binding_t*)iree_alloca(
          binding_count * sizeof(iree_hal_descriptor_set_binding_t));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_check_deref(args->a3[i].r1, &bindings[i].buffer));
    bindings[i].binding = (uint32_t)args->a3[i].i0;
    bindings[i].offset = (iree_device_size_t)args->a3[i].i2;
    bindings[i].length = (iree_device_size_t)args->a3[i].i3;
  }

  return iree_hal_command_buffer_push_descriptor_set(
      command_buffer, executable_layout, set, binding_count, bindings);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_bind_descriptor_set,  //
                   iree_hal_module_state_t,                             //
                   rrirCiD, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_layout_check_deref(args->r1, &executable_layout));
  int32_t set = args->i2;
  iree_hal_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_descriptor_set_check_deref(args->r3, &descriptor_set));
  iree_host_size_t dynamic_offset_count = 0;
  iree_device_size_t* dynamic_offsets = NULL;
  IREE_VM_ABI_VLA_STACK_CAST(args, a4_count, a4, iree_device_size_t, 64,
                             &dynamic_offset_count, &dynamic_offsets);
  return iree_hal_command_buffer_bind_descriptor_set(
      command_buffer, executable_layout, set, descriptor_set,
      dynamic_offset_count, dynamic_offsets);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_dispatch,  //
                   iree_hal_module_state_t,                  //
                   rriiii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_check_deref(args->r1, &executable));
  uint32_t entry_point = (uint32_t)args->i2;
  uint32_t workgroup_x = (uint32_t)args->i3;
  uint32_t workgroup_y = (uint32_t)args->i4;
  uint32_t workgroup_z = (uint32_t)args->i5;
  return iree_hal_command_buffer_dispatch(command_buffer, executable,
                                          entry_point, workgroup_x, workgroup_y,
                                          workgroup_z);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_dispatch_indirect,  //
                   iree_hal_module_state_t,                           //
                   rriri, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_check_deref(args->r1, &executable));
  uint32_t entry_point = (uint32_t)args->i2;
  iree_hal_buffer_t* workgroups_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref(args->r3, &workgroups_buffer));
  iree_vm_size_t workgroups_offset = (iree_vm_size_t)args->i4;
  return iree_hal_command_buffer_dispatch_indirect(
      command_buffer, executable, entry_point, workgroups_buffer,
      workgroups_offset);
}

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_descriptor_set_create,  //
                   iree_hal_module_state_t,                //
                   rrCiriiD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_descriptor_set_layout_t* set_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_descriptor_set_layout_check_deref(args->r1, &set_layout));

  iree_host_size_t binding_count = args->a2_count;
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "binding count %zu > %zu",
                            binding_count,
                            IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_descriptor_set_binding_t* bindings =
      (iree_hal_descriptor_set_binding_t*)iree_alloca(
          binding_count * sizeof(iree_hal_descriptor_set_binding_t));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_check_deref(args->a2[i].r1, &bindings[i].buffer));
    bindings[i].binding = (uint32_t)args->a2[i].i0;
    bindings[i].offset = (iree_device_size_t)args->a2[i].i2;
    bindings[i].length = (iree_device_size_t)args->a2[i].i3;
  }

  iree_hal_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_create(
      device, set_layout, binding_count, bindings, &descriptor_set));
  rets->r0 = iree_hal_descriptor_set_move_ref(descriptor_set);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_layout
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_descriptor_set_layout_create,  //
                   iree_hal_module_state_t,                       //
                   riCiiD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_descriptor_set_layout_usage_type_t usage_type =
      (iree_hal_descriptor_set_layout_usage_type_t)args->i1;

  iree_host_size_t binding_count = args->a2_count;
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "binding count %zu > %zu",
                            binding_count,
                            IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_descriptor_set_layout_binding_t* bindings =
      (iree_hal_descriptor_set_layout_binding_t*)iree_alloca(
          binding_count * sizeof(iree_hal_descriptor_set_layout_binding_t));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    bindings[i].binding = (uint32_t)args->a2[i].i0;
    bindings[i].type = (iree_hal_descriptor_type_t)args->a2[i].i1;
  }

  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_layout_create(
      device, usage_type, binding_count, bindings, &descriptor_set_layout));
  rets->r0 = iree_hal_descriptor_set_layout_move_ref(descriptor_set_layout);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_device_allocator,  //
                   iree_hal_module_state_t,           //
                   r, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  rets->r0 = iree_hal_allocator_retain_ref(iree_hal_device_allocator(device));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_query_i32,  //
                   iree_hal_module_state_t,           //
                   rrr, ii) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_vm_buffer_t* category = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &category));
  iree_string_view_t category_str = iree_vm_buffer_as_string(category);
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r2, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  int32_t value = 0;
  iree_status_t query_status =
      iree_hal_device_query_i32(device, category_str, key_str, &value);
  rets->i0 = iree_status_consume_code(query_status) == IREE_STATUS_OK ? 1 : 0;
  rets->i1 = (int32_t)value;
  return iree_ok_status();
}

//===--------------------------------------------------------------------===//
// iree_hal_executable_t
//===--------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_executable_create,  //
                   iree_hal_module_state_t,            //
                   rrrrCrD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_vm_buffer_t* executable_format = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref(args->r1, &executable_format));
  iree_string_view_t executable_format_str =
      iree_vm_buffer_as_string(executable_format);
  iree_vm_buffer_t* executable_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r2, &executable_data));
  iree_host_size_t constant_count = 0;
  const uint32_t* constants = NULL;
  if (iree_vm_buffer_isa(args->r3)) {
    iree_vm_buffer_t* constant_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_check_deref(args->r3, &constant_buffer));
    if (constant_buffer->data.data_length % 4 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant buffer data must contain 4-byte "
                              "elements but data length is %" PRIhsz,
                              constant_buffer->data.data_length);
    }
    constant_count = constant_buffer->data.data_length / sizeof(uint32_t);
    constants = (const uint32_t*)constant_buffer->data.data;
  }
  iree_host_size_t executable_layout_count = args->a4_count;
  iree_hal_executable_layout_t** executable_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator,
      executable_layout_count * sizeof(executable_layouts[0]),
      (void**)&executable_layouts));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < executable_layout_count; ++i) {
    status = iree_hal_executable_layout_check_deref(args->a4[i].r0,
                                                    &executable_layouts[i]);
    if (!iree_status_is_ok(status)) break;
  }

  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode |=
        executable_data->access == IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE
            ? IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA
            : 0;
    executable_params.executable_format = executable_format_str;
    executable_params.executable_data = iree_make_const_byte_span(
        executable_data->data.data, executable_data->data.data_length);
    executable_params.executable_layout_count = executable_layout_count;
    executable_params.executable_layouts = executable_layouts;
    executable_params.constant_count = constant_count;
    executable_params.constants = constants;
    status = iree_hal_executable_cache_prepare_executable(
        state->executable_cache, &executable_params, &executable);
  }

  iree_allocator_free(state->host_allocator, executable_layouts);
  rets->r0 = iree_hal_executable_move_ref(executable);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_executable_layout_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_executable_layout_create,  //
                   iree_hal_module_state_t,                   //
                   riCrD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  int32_t push_constants = (int32_t)args->i1;
  iree_host_size_t set_layout_count = 0;
  iree_hal_descriptor_set_layout_t** set_layouts = NULL;
  IREE_VM_ABI_VLA_STACK_DEREF(args, a2_count, a2,
                              iree_hal_descriptor_set_layout, 32,
                              &set_layout_count, &set_layouts);

  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_layout_create(
      device, push_constants, set_layout_count, set_layouts,
      &executable_layout));
  rets->r0 = iree_hal_executable_layout_move_ref(executable_layout);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_create,  //
                   iree_hal_module_state_t,           //
                   ri, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  uint32_t initial_value = (uint32_t)args->i1;

  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device, initial_value, &semaphore));
  rets->r0 = iree_hal_semaphore_move_ref(semaphore);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_query,  //
                   iree_hal_module_state_t,          //
                   r, ii) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));

  uint64_t value = 0;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  rets->i0 = iree_status_consume_code(query_status);
  rets->i1 = (uint32_t)value;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_signal,  //
                   iree_hal_module_state_t,           //
                   ri, v) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
  uint32_t new_value = (uint32_t)args->i1;

  return iree_hal_semaphore_signal(semaphore, new_value);
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_fail,  //
                   iree_hal_module_state_t,         //
                   ri, v) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
  iree_status_code_t status_code =
      (iree_status_code_t)(args->i1 & IREE_STATUS_CODE_MASK);

  iree_hal_semaphore_fail(semaphore, iree_make_status(status_code));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_await,  //
                   iree_hal_module_state_t,          //
                   ri, i) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
  uint64_t new_value = (uint32_t)args->i1;

  // TODO(benvanik): coroutine magic.
  iree_status_t status =
      iree_hal_semaphore_wait(semaphore, new_value, iree_infinite_timeout());
  if (iree_status_is_ok(status)) {
    rets->i0 = 0;
  } else if (iree_status_is_deadline_exceeded(status)) {
    // Propagate deadline exceeded back to the VM.
    rets->i0 = (int32_t)iree_status_consume_code(status);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_hal_module_exports_ table.
static const iree_vm_native_function_ptr_t iree_hal_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/hal/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_hal_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_hal_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .reflection_attr_count = 0,                                  \
      .reflection_attrs = NULL,                                    \
  },
#include "iree/modules/hal/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_hal_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_hal_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_hal_module_descriptor_ = {
    .module_name = iree_string_view_literal("hal"),
    .import_count = 0,  // workaround for 0-length C struct
    .imports = iree_hal_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_hal_module_exports_),
    .exports = iree_hal_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_hal_module_funcs_),
    .functions = iree_hal_module_funcs_,
    .reflection_attr_count = 0,
    .reflection_attrs = NULL,
};

IREE_API_EXPORT iree_status_t
iree_hal_module_create(iree_hal_device_t* device, iree_allocator_t allocator,
                       iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_hal_module_destroy,
      .alloc_state = iree_hal_module_alloc_state,
      .free_state = iree_hal_module_free_state,
      .notify = iree_hal_module_notify,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_hal_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_hal_module_descriptor_, allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, base_module);
    return status;
  }

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);
  module->host_allocator = allocator;
  module->shared_device = device;
  iree_hal_device_retain(module->shared_device);

  *out_module = base_module;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_device_t* iree_hal_module_state_device(
    iree_vm_module_state_t* module_state) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  return state->shared_device;
}

//===--------------------------------------------------------------------===//
// Utilities
//===--------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_assign(
    const iree_vm_list_t* list, iree_host_size_t i) {
  return (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
      list, i, iree_hal_buffer_view_get_descriptor());
}

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_retain(
    const iree_vm_list_t* list, iree_host_size_t i) {
  iree_hal_buffer_view_t* value = iree_vm_list_get_buffer_view_assign(list, i);
  iree_hal_buffer_view_retain(value);
  return value;
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_buffer_view_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_buffer_view_t* value) {
  iree_vm_ref_t value_ref;
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
      value, iree_hal_buffer_view_type_id(), &value_ref));
  return iree_vm_list_set_ref_retain(list, i, &value_ref);
}
