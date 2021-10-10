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
                              iree_hal_buffer_destroy,
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
  iree_hal_executable_cache_t* executable_cache;

  iree_hal_semaphore_t* submit_semaphore;
  uint64_t submit_value;

  void* deferred_lru[6];
  iree_vm_list_t* deferred_releases;
} iree_hal_module_state_t;

static void IREE_API_PTR iree_hal_module_destroy(void* base_module) {
  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);
  iree_hal_device_release(module->shared_device);
}

static iree_status_t IREE_API_PTR
iree_hal_module_alloc_state(void* self, iree_allocator_t host_allocator,
                            iree_vm_module_state_t** out_module_state) {
  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(self);
  iree_hal_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->shared_device = module->shared_device;
  iree_hal_device_retain(state->shared_device);

  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      /*element_type=*/NULL, /*initial_capacity=*/512, state->host_allocator,
      &state->deferred_releases));

  IREE_RETURN_IF_ERROR(iree_hal_executable_cache_create(
      state->shared_device, iree_string_view_empty(),
      &state->executable_cache));

  state->submit_value = 0ull;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      state->shared_device, state->submit_value, &state->submit_semaphore));

  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR
iree_hal_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  iree_hal_semaphore_release(state->submit_semaphore);
  iree_vm_list_release(state->deferred_releases);
  iree_hal_executable_cache_release(state->executable_cache);
  iree_hal_device_release(state->shared_device);
  iree_allocator_free(state->host_allocator, state);
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

void iree_hal_module_ex_defer_release(iree_hal_module_state_t* state,
                                      const iree_vm_ref_t value) {
  // A bulk of the calls to this are for the same (or very recently same)
  // objects, such as constant pool or transient buffer storage that may be
  // bound 4-10 times per dispatch. This tiny LRU lets us avoid adding such
  // repeated patterns in the common case.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(state->deferred_lru); ++i) {
    if (state->deferred_lru[i] == value.ptr) {
      // Hit - keep the list sorted by most->least recently used.
      state->deferred_lru[i] = state->deferred_lru[0];
      state->deferred_lru[0] = value.ptr;
      return;
    }
  }
  // Miss - shift the list down and insert the new item at the head.
  memmove(&state->deferred_lru[1], &state->deferred_lru[0],
          sizeof(state->deferred_lru[0]) *
              (IREE_ARRAYSIZE(state->deferred_lru) - 1));
  state->deferred_lru[0] = value.ptr;

  IREE_IGNORE_ERROR(
      iree_vm_list_push_ref_retain(state->deferred_releases, &value));
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

  // Drop all pending deferred releases (references to everything in flight).
  // This will be replaced with resource sets in the future that are attached to
  // each command buffer.
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(state->deferred_releases, 0));
  memset(state->deferred_lru, 0, sizeof(state->deferred_lru));

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

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, memory_types, buffer_usage, allocation_size, &buffer));
  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

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

  // TODO(benvanik): wrap when supported.

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

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(allocator, memory_types, buffer_usage,
                                         length, &buffer),
      "failed to allocate buffer of length %d", length);

  iree_status_t status =
      iree_hal_buffer_write_data(buffer, 0, source->data.data + offset, length);
  if (iree_status_is_ok(status)) {
    rets->r0 = iree_hal_buffer_move_ref(buffer);
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_allocator,  //
                   iree_hal_module_state_t,           //
                   r, r) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->r0 = iree_hal_allocator_retain_ref(iree_hal_buffer_allocator(buffer));
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

  IREE_RETURN_IF_ERROR(iree_hal_buffer_read_data(source_buffer, source_offset,
                                                 &target_buffer, length));

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

  return iree_hal_buffer_write_data(target_buffer, target_offset, &value,
                                    length);
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
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_create(source_buffer, shape_dims, shape_rank,
                                  element_type, encoding_type, &buffer_view));
  rets->r0 = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_ok_status();
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
                   rriii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &target_buffer));
  iree_vm_size_t target_offset = (iree_vm_size_t)args->i2;
  iree_vm_size_t length = (iree_vm_size_t)args->i3;
  uint32_t pattern = (uint32_t)args->i4;

  iree_hal_module_ex_defer_release(state, args->r1);

  return iree_hal_command_buffer_fill_buffer(command_buffer, target_buffer,
                                             target_offset, length, &pattern,
                                             sizeof(pattern));
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

  iree_hal_module_ex_defer_release(state, args->r1);
  iree_hal_module_ex_defer_release(state, args->r3);

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
    iree_hal_module_ex_defer_release(state, args->a3[i].r1);
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

  iree_hal_module_ex_defer_release(state, args->r3);

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

  iree_hal_module_ex_defer_release(state, args->r1);

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

  iree_hal_module_ex_defer_release(state, args->r1);
  iree_hal_module_ex_defer_release(state, args->r3);

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
                   riCiiiD, r) {
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
    bindings[i].access = (iree_hal_memory_access_t)args->a2[i].i2;
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
                   rrrCrD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_vm_buffer_t* executable_format = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref(args->r1, &executable_format));
  iree_string_view_t executable_format_str =
      iree_vm_buffer_as_string(executable_format);
  iree_vm_buffer_t* executable_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r2, &executable_data));
  iree_host_size_t executable_layout_count = args->a3_count;
  iree_hal_executable_layout_t** executable_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator,
      executable_layout_count * sizeof(executable_layouts[0]),
      (void**)&executable_layouts));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < executable_layout_count; ++i) {
    status = iree_hal_executable_layout_check_deref(args->a3[i].r0,
                                                    &executable_layouts[i]);
    if (!iree_status_is_ok(status)) break;
  }

  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_executable_spec_t spec;
    iree_hal_executable_spec_initialize(&spec);
    spec.caching_mode |=
        executable_data->access == IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE
            ? IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA
            : 0;
    spec.executable_format = executable_format_str;
    spec.executable_data = iree_make_const_byte_span(
        executable_data->data.data, executable_data->data.data_length);
    spec.executable_layout_count = executable_layout_count;
    spec.executable_layouts = executable_layouts;
    status = iree_hal_executable_cache_prepare_executable(
        state->executable_cache, &spec, &executable);
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
