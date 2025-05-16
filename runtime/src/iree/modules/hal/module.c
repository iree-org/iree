// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/module.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include "iree/modules/hal/utils/buffer_diagnostics.h"

//===----------------------------------------------------------------------===//
// Limits imposed by the module (and not the HAL)
//===----------------------------------------------------------------------===//

// Limit the number of bindings we pass down through the HAL. This can be tuned
// in the future but right now guards the stack from blowing up during calls.
#define IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT ((iree_host_size_t)32)

// Limit the number of bindings in a binding table that we allocate on the stack
// while marshaling from the VM. Counts over this amount will result in heap
// allocations to avoid blowing the native stack. In most programs we expect
// at most a dozen buffers but programs with individually stored parameters may
// need hundreds or even thousands. Yuck.
#define IREE_HAL_MODULE_MAX_STACK_COMMAND_BUFFER_BINDING_COUNT \
  ((iree_host_size_t)64)

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

#define IREE_HAL_MODULE_VERSION_0_6 0x00000006u
#define IREE_HAL_MODULE_VERSION_LATEST IREE_HAL_MODULE_VERSION_0_6

typedef struct iree_hal_module_t {
  iree_allocator_t host_allocator;
  iree_hal_module_flags_t flags;
  iree_hal_module_debug_sink_t debug_sink;
  iree_host_size_t device_count;
  iree_hal_device_t* devices[];
} iree_hal_module_t;

#define IREE_HAL_MODULE_CAST(module) \
  (iree_hal_module_t*)((uint8_t*)(module) + iree_vm_native_module_size());

static void IREE_API_PTR iree_hal_module_destroy(void* base_module) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);

  // Release the debug sink prior to releasing devices as it may be caching
  // device-specific information that will be unavailable once the devices are
  // released.
  if (module->debug_sink.release.fn) {
    module->debug_sink.release.fn(module->debug_sink.release.user_data);
  }

  // Release all devices. The module may be the last retainer and the devices
  // (and their corresponding drivers) may be immediately unloaded.
  for (iree_host_size_t i = 0; i < module->device_count; ++i) {
    iree_hal_device_release(module->devices[i]);
  }

  IREE_TRACE_ZONE_END(z0);
}

typedef struct iree_hal_module_state_t {
  iree_allocator_t host_allocator;

  // Flags controlling HAL module behavior passed in from the hosting
  // application. All instantiations of a module share the same flags.
  iree_hal_module_flags_t flags;

  // Debug sink for routing debug events.
  iree_hal_module_debug_sink_t debug_sink;

  // Total number of devices available to the module.
  iree_host_size_t device_count;
  // Devices referencing the storage in the parent module.
  // Unretained as the parent module must remain live longer than any module
  // state allocated from it and we can rely on it to keep the devices retained.
  iree_hal_device_t** devices;

  // TODO(benvanik): add iree_loop_t to module constructor.
  // Status of the nested loop we run for executable creation today. We should
  // instead be taking a loop upon creation and scheduling work against that.
  iree_status_t loop_status;

  // Shared executable cache for each device used to cache all executables
  // created in the context. We could have multiple to allow for modules to
  // create distinct sets of executables like ones for training vs inference in
  // the same model or allow these to be injected so that multiple loaded
  // contexts share the caches.
  iree_hal_executable_cache_t* executable_caches[];
} iree_hal_module_state_t;

static iree_status_t IREE_API_PTR
iree_hal_module_alloc_state(void* self, iree_allocator_t host_allocator,
                            iree_vm_module_state_t** out_module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(self);
  iree_hal_module_state_t* state = NULL;
  iree_host_size_t total_size =
      sizeof(*state) +
      module->device_count * sizeof(state->executable_caches[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&state));
  memset(state, 0, total_size);
  state->host_allocator = host_allocator;
  state->flags = module->flags;
  state->debug_sink = module->debug_sink;
  state->device_count = module->device_count;
  state->devices = module->devices;
  state->loop_status = iree_ok_status();

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < state->device_count; ++i) {
    status = iree_hal_executable_cache_create(
        state->devices[i], iree_string_view_empty(),
        iree_loop_inline(&state->loop_status), &state->executable_caches[i]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_module_state = (iree_vm_module_state_t*)state;
  } else {
    for (iree_host_size_t i = 0; i < state->device_count; ++i) {
      iree_hal_executable_cache_release(state->executable_caches[i]);
    }
    iree_allocator_free(host_allocator, state);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void IREE_API_PTR
iree_hal_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  for (iree_host_size_t i = 0; i < state->device_count; ++i) {
    iree_hal_executable_cache_release(state->executable_caches[i]);
  }
  iree_status_ignore(state->loop_status);
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_hal_module_fork_state(
    void* self, iree_vm_module_state_t* base_parent_state,
    iree_allocator_t host_allocator, iree_vm_module_state_t** out_child_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_child_state = NULL;

  iree_hal_module_state_t* parent_state =
      (iree_hal_module_state_t*)base_parent_state;

  // The base module state is derived entirely from the shared module.
  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(self);
  iree_hal_module_state_t* child_state = NULL;
  iree_host_size_t total_size =
      sizeof(*child_state) +
      module->device_count * sizeof(child_state->executable_caches[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&child_state));
  memset(child_state, 0, total_size);
  child_state->host_allocator = host_allocator;
  child_state->flags = module->flags;
  child_state->device_count = module->device_count;
  child_state->devices = module->devices;
  child_state->loop_status = iree_ok_status();

  // Reference the parent executable caches.
  for (iree_host_size_t i = 0; i < child_state->device_count; ++i) {
    iree_hal_executable_cache_t* executable_cache =
        parent_state->executable_caches[i];
    child_state->executable_caches[i] = executable_cache;
    iree_hal_executable_cache_retain(executable_cache);
  }

  *out_child_state = (iree_vm_module_state_t*)child_state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Returns an unretained reference to the executable cache for the given device.
// If the same device is registered multiple times the first cache is returned.
static iree_status_t iree_hal_module_state_lookup_executable_cache(
    iree_hal_module_state_t* state, iree_hal_device_t* device,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  for (iree_host_size_t i = 0; i < state->device_count; ++i) {
    if (state->devices[i] == device) {
      *out_executable_cache = state->executable_caches[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "no executable cache for the given device found; possibly a device not "
      "registered with the HAL module");
}

static iree_status_t IREE_API_PTR iree_hal_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  switch (signal) {
    case IREE_VM_SIGNAL_SUSPEND:
    case IREE_VM_SIGNAL_LOW_MEMORY: {
      for (iree_host_size_t i = 0; i < state->device_count; ++i) {
        IREE_RETURN_IF_ERROR(iree_hal_device_trim(state->devices[i]));
      }
      return iree_ok_status();
    }
    default: {
      // Ignored today but if we started managing device power down we could
      // use this to wake them back up again.
      return iree_ok_status();
    }
  }
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Casts a VM value to a C host size.
static iree_host_size_t iree_hal_cast_host_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if host
  // size is 32-bits.
  return (iree_host_size_t)value;
}

// Casts a VM value to a HAL device size.
static iree_device_size_t iree_hal_cast_device_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if device
  // size is 32-bits.
  return (iree_device_size_t)value;
}

//===----------------------------------------------------------------------===//
// Experimental APIs
//===----------------------------------------------------------------------===//
// NOTE: Ex* APIs are experimental and likely to be removed soon. Modules
// using these APIs are not forward compatible.

static void iree_hal_module_file_buffer_release(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_vm_buffer_t* backing_buffer = (iree_vm_buffer_t*)user_data;
  iree_vm_buffer_release(backing_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_ex_file_from_memory,  //
                   iree_hal_module_state_t,              //
                   rIirIIi, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_memory_access_t access = (iree_hal_memory_access_t)args->i2;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r3, &buffer));
  iree_host_size_t offset = iree_hal_cast_host_size(args->i4);
  iree_host_size_t length = iree_hal_cast_host_size(args->i5);
  uint32_t flags = (uint32_t)args->i6;

  // Only allow read-only access right now while experimental.
  // The contents here are almost always from mapped file memory today.
  if (iree_any_bit_set(access, ~IREE_HAL_MEMORY_ACCESS_READ)) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "only read-only memory can be accessed via a file handle (today)");
  }

  // Verify the provided range and get the host pointer.
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(buffer, offset, length, 1, &span));

  // Retain the buffer until the file is destroyed.
  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_hal_module_file_buffer_release,
      .user_data = buffer,
  };
  iree_vm_buffer_retain(buffer);

  // Wrap the memory in a file handle.
  iree_io_file_handle_t* handle = NULL;
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span((void*)span.data, span.data_length), release_callback,
      iree_hal_device_host_allocator(device), &handle);
  if (!iree_status_is_ok(status)) {
    iree_vm_buffer_release(buffer);
  }

  // Attempt to import the memory as a file.
  // Memory files are always supported (even if via emulation) so this should
  // always succeed.
  iree_hal_file_t* file = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(device, queue_affinity, access, handle, flags,
                                  &file);
  }

  iree_io_file_handle_release(handle);

  rets->r0 = iree_hal_file_move_ref(file);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_allocate,  //
                   iree_hal_module_state_t,             //
                   rIiiI, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i2;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i3;
  iree_device_size_t allocation_size = iree_hal_cast_device_size(args->i4);

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
      .queue_affinity = queue_affinity,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
                           allocator, params, allocation_size, &buffer),
                       "failed to allocate buffer of length %" PRIdsz,
                       allocation_size);

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

static void iree_hal_module_imported_buffer_release(void* user_data,
                                                    iree_hal_buffer_t* buffer) {
  iree_vm_buffer_t* backing_buffer = (iree_vm_buffer_t*)user_data;
  iree_vm_buffer_release(backing_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_import,  //
                   iree_hal_module_state_t,           //
                   riIiirII, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  bool is_try = args->i1 != 0;
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i2;
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i3;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i4;
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r5, &source));
  iree_device_size_t offset = iree_hal_cast_device_size(args->i6);
  iree_device_size_t length = iree_hal_cast_device_size(args->i7);

  iree_host_size_t buffer_length = source->data.data_length;
  if (length == -1) {
    length = buffer_length;
  }
  if (length < 0 || offset < 0 || offset > buffer_length ||
      offset + length > buffer_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte range out of bounds (requested %" PRIdsz
                            "-%" PRIdsz " of available %" PRIhsz ")",
                            offset, (offset + length - 1), buffer_length);
  }

  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_READ;
  if (!iree_all_bits_set(source->access, IREE_VM_BUFFER_ACCESS_MUTABLE)) {
    // Source buffer is read-only; require that the access request matches.
    if (!iree_all_bits_set(buffer_usage,
                           IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE)) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "source buffer is immutable and can only be "
                              "imported for constant usage");
    }

    // NOTE: if we wanted to lock things down for when there's no MMU to ensure
    // that the loaded program doesn't touch the memory then we could just fail
    // the request - the program will then perform an alloc+copy and can do
    // whatever it wants with the memory.
  } else {
    // Source buffer is mutable; allow in-place writes.
    if (!iree_all_bits_set(buffer_usage,
                           IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE)) {
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
      .queue_affinity = queue_affinity,
  };
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = length,
      .handle.host_allocation.ptr = source->data.data + offset,
  };
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_module_imported_buffer_release,
      .user_data = source,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer, release_callback, &buffer);
  if (iree_status_is_ok(status)) {
    // Import succeeded - retain the source buffer that'll be released by
    // iree_hal_module_map_data_ctl when the mapping is no longer used.
    iree_vm_buffer_retain(source);
    rets->r0 = iree_hal_buffer_move_ref(buffer);
    return iree_ok_status();
  }

  // Failed to import - if this was a try then don't fail and just rely on the
  // result being nullptr to indicate to the caller that things failed.
  memset(&rets->r0, 0, sizeof(rets->r0));
  if (is_try) {
    IREE_TRACE_MESSAGE(WARNING, "try import failed");
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_assert,  //
                   iree_hal_module_state_t,        //
                   rrrIii, v) {
  IREE_RETURN_IF_ERROR(iree_hal_modules_buffer_assert(
      args->r0, args->r1, iree_hal_cast_device_size(args->i3),
      (iree_hal_memory_type_t)args->i4, (iree_hal_buffer_usage_t)args->i5));

  // TODO(benvanik): assert that the buffer view is accessible from the
  // target device. This needs some iree_hal_allocator_* methods for checking
  // whether the external buffer can be used. To start we just compare if the
  // allocators are identical.
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r2, &allocator));
  (void)allocator;

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_allocation_preserve,  //
                   iree_hal_module_state_t,                     //
                   r, v) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  iree_hal_buffer_allocation_preserve(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_allocation_discard,  //
                   iree_hal_module_state_t,                    //
                   r, i) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->i0 = iree_hal_buffer_allocation_discard(buffer) ? 1 : 0;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_allocation_is_terminal,  //
                   iree_hal_module_state_t,                        //
                   r, i) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->i0 = iree_hal_buffer_allocation_is_terminal(buffer) ? 1 : 0;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_subspan,  //
                   iree_hal_module_state_t,         //
                   rII, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_device_size_t length = iree_hal_cast_device_size(args->i2);

  iree_hal_buffer_t* subspan_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_subspan(source_buffer, source_offset, length,
                              state->host_allocator, &subspan_buffer),
      "invalid subspan of an existing buffer (source_offset=%" PRIdsz
      ", length=%" PRIdsz ")",
      source_offset, length);
  rets->r0 = iree_hal_buffer_move_ref(subspan_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_length,  //
                   iree_hal_module_state_t,        //
                   r, I) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->i0 = (int64_t)iree_hal_buffer_byte_length(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_load,  //
                   iree_hal_module_state_t,      //
                   rIi, i) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_vm_size_t length = (iree_vm_size_t)args->i2;

  uint32_t target_buffer = 0;
  if (length > sizeof(target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "load length byte count %d exceeds max", length);
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(source_buffer, source_offset,
                                                &target_buffer, length));

  rets->i0 = target_buffer;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_store,  //
                   iree_hal_module_state_t,       //
                   irIi, v) {
  int32_t value = args->i0;
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i2);
  iree_vm_size_t length = (iree_vm_size_t)args->i3;

  if (length > sizeof(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "store length byte count %d exceeds max", length);
  } else if (target_offset + length >
             iree_hal_buffer_byte_length(target_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "store out of bounds (target_offset=%" PRIdsz
                            ", length=%d into max %" PRIdsz ")",
                            target_offset, length,
                            iree_hal_buffer_byte_length(target_buffer));
  }

  return iree_hal_buffer_map_write(target_buffer, target_offset, &value,
                                   length);
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_create,  //
                   iree_hal_module_state_t,             //
                   rIIiiCID, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_device_size_t source_length = iree_hal_cast_device_size(args->i2);
  iree_hal_element_type_t element_type = (iree_hal_element_type_t)args->i3;
  iree_hal_encoding_type_t encoding_type = (iree_hal_encoding_type_t)args->i4;
  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t* shape_dims = NULL;
  // TODO(benvanik): avoid the cast/alloca if not required.
  IREE_VM_ABI_VLA_STACK_CAST(args, a5_count, a5, iree_hal_dim_t, 128,
                             &shape_rank, &shape_dims);

  iree_hal_buffer_t* subspan_buffer = NULL;
  if (source_offset != 0 ||
      source_length != iree_hal_buffer_byte_length(source_buffer)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_subspan(source_buffer, source_offset, source_length,
                                state->host_allocator, &subspan_buffer),
        "invalid subspan of an existing buffer (source_offset=%" PRIdsz
        ", length=%" PRIdsz ")",
        source_offset, source_length);
  }

  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      subspan_buffer ? subspan_buffer : source_buffer, shape_rank, shape_dims,
      element_type, encoding_type, state->host_allocator, &buffer_view));

  iree_hal_buffer_release(subspan_buffer);

  rets->r0 = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_assert,  //
                   iree_hal_module_state_t,             //
                   rriiCID, v) {
  iree_host_size_t expected_shape_rank = 0;
  iree_hal_dim_t* expected_shape_dims = NULL;
  // TODO(benvanik): avoid the cast/alloca if not required.
  IREE_VM_ABI_VLA_STACK_CAST(args, a4_count, a4, iree_hal_dim_t, 128,
                             &expected_shape_rank, &expected_shape_dims);
  return iree_hal_modules_buffer_view_assert(
      args->r0, args->r1, (iree_hal_element_type_t)args->i2,
      (iree_hal_encoding_type_t)args->i3, expected_shape_rank,
      expected_shape_dims);
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
                   ri, I) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  iree_vm_size_t index = (iree_vm_size_t)args->i1;
  rets->i0 = (int64_t)iree_hal_buffer_view_shape_dim(buffer_view, index);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_trace,  //
                   iree_hal_module_state_t,            //
                   rCrD, v) {
  if (state->debug_sink.buffer_view_trace.fn) {
    iree_vm_buffer_t* key = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r0, &key));
    iree_string_view_t key_str = iree_vm_buffer_as_string(key);
    iree_host_size_t buffer_view_count =
        iree_hal_cast_host_size(args->a1_count);
    if (buffer_view_count > 128) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many buffer views for a single trace call");
    }
    iree_hal_buffer_view_t** buffer_views =
        iree_alloca(buffer_view_count * sizeof(iree_hal_buffer_view_t*));
    for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_check_deref(args->a1[i].r0, &buffer_views[i]));
    }
    return state->debug_sink.buffer_view_trace.fn(
        state->debug_sink.buffer_view_trace.user_data, key_str,
        buffer_view_count, buffer_views, state->host_allocator);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_channel_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_channel_create,  //
                   iree_hal_module_state_t,         //
                   rIIrrii, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_channel_flags_t flags = (iree_hal_channel_flags_t)args->i2;
  iree_vm_buffer_t* id = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref_or_null(args->r3, &id));
  iree_vm_buffer_t* group = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref_or_null(args->r4, &group));
  iree_string_view_t group_str = iree_vm_buffer_as_string(group);
  int32_t rank = args->i5;
  int32_t count = args->i6;

  iree_hal_channel_params_t params = {
      .flags = flags,
      .id = iree_vm_buffer_const_contents(id),  // may be null
      .group = group_str,                       // may be null
      .rank = rank,
      .count = count,
  };

  iree_hal_channel_t* channel = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_channel_create(device, queue_affinity, params, &channel));

  rets->r0 = iree_hal_channel_move_ref(channel);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_channel_split,  //
                   iree_hal_module_state_t,        //
                   riiI, r) {
  iree_hal_channel_t* base_channel = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_channel_check_deref(args->r0, &base_channel));
  int32_t color = args->i1;
  int32_t key = args->i2;
  iree_hal_channel_flags_t flags = (iree_hal_channel_flags_t)args->i3;

  iree_hal_channel_t* split_channel = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_channel_split(base_channel, color, key, flags, &split_channel));

  rets->r0 = iree_hal_channel_move_ref(split_channel);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_channel_rank_and_count,  //
                   iree_hal_module_state_t,                 //
                   r, ii) {
  iree_hal_channel_t* channel = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_channel_check_deref(args->r0, &channel));

  int32_t rank = 0;
  int32_t count = 0;
  iree_hal_channel_query_rank_and_count(channel, &rank, &count);

  rets->i0 = rank;
  rets->i1 = count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_create,  //
                   iree_hal_module_state_t,                //
                   riiIi, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_command_buffer_mode_t modes =
      (iree_hal_command_buffer_mode_t)args->i1;
  iree_hal_command_category_t command_categories =
      (iree_hal_command_category_t)args->i2;
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i3;
  iree_host_size_t binding_capacity = (iree_host_size_t)args->i4;

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device, modes, command_categories, queue_affinity, binding_capacity,
      &command_buffer));

  iree_status_t status = iree_hal_command_buffer_begin(command_buffer);
  if (iree_status_is_ok(status)) {
    rets->r0 = iree_hal_command_buffer_move_ref(command_buffer);
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_finalize,  //
                   iree_hal_module_state_t,                  //
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
  return iree_hal_command_buffer_begin_debug_group(
      command_buffer, label_str, iree_hal_label_color_unspecified(), &location);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_end_debug_group,  //
                   iree_hal_module_state_t,                         //
                   r, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  return iree_hal_command_buffer_end_debug_group(command_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_execution_barrier,  //
                   iree_hal_module_state_t,                           //
                   riiI, v) {
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

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_advise_buffer,  //
                   iree_hal_module_state_t,                       //
                   rrIIIi, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  uint32_t buffer_slot = (uint32_t)args->i5;
  iree_hal_buffer_ref_t buffer_ref =
      iree_hal_make_indirect_buffer_ref(buffer_slot, 0, IREE_HAL_WHOLE_BUFFER);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r1, &buffer_ref.buffer));
  iree_hal_memory_advise_flags_t flags =
      (iree_hal_memory_advise_flags_t)args->i2;
  uint64_t arg0 = (uint64_t)args->i3;
  uint64_t arg1 = (uint64_t)args->i4;
  return iree_hal_command_buffer_advise_buffer(command_buffer, buffer_ref,
                                               flags, arg0, arg1);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_fill_buffer,  //
                   iree_hal_module_state_t,                     //
                   rrIIiIiI, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i2);
  iree_device_size_t length = iree_hal_cast_device_size(args->i3);
  uint32_t target_buffer_slot = (uint32_t)args->i4;
  iree_hal_buffer_ref_t target_ref = iree_hal_make_indirect_buffer_ref(
      target_buffer_slot, target_offset, length);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r1, &target_ref.buffer));
  uint64_t pattern = (uint64_t)args->i5;
  uint32_t pattern_length = (uint32_t)args->i6;
  iree_hal_fill_flags_t flags = (iree_hal_fill_flags_t)args->i7;
  return iree_hal_command_buffer_fill_buffer(command_buffer, target_ref,
                                             &pattern, pattern_length, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_update_buffer,  //
                   iree_hal_module_state_t,                       //
                   rrIrIIiI, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_vm_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &source_buffer));
  iree_host_size_t source_offset = iree_hal_cast_host_size(args->i2);
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i4);
  iree_device_size_t length = iree_hal_cast_device_size(args->i5);
  uint32_t target_buffer_slot = (uint32_t)args->i6;
  iree_hal_update_flags_t flags = (iree_hal_update_flags_t)args->i7;
  iree_hal_buffer_ref_t target_ref = iree_hal_make_indirect_buffer_ref(
      target_buffer_slot, target_offset, length);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r3, &target_ref.buffer));
  iree_const_byte_span_t source_span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(
      source_buffer, source_offset, (iree_host_size_t)length, 1, &source_span));
  return iree_hal_command_buffer_update_buffer(command_buffer, source_span.data,
                                               /*source_offset=*/0, target_ref,
                                               flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_copy_buffer,  //
                   iree_hal_module_state_t,                     //
                   riirIrIII, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  uint32_t source_buffer_slot = (uint32_t)args->i1;
  uint32_t target_buffer_slot = (uint32_t)args->i2;
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i4);
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i6);
  iree_device_size_t length = iree_hal_cast_device_size(args->i7);
  iree_hal_copy_flags_t flags = (iree_hal_copy_flags_t)args->i8;
  iree_hal_buffer_ref_t source_ref = iree_hal_make_indirect_buffer_ref(
      source_buffer_slot, source_offset, length);
  iree_hal_buffer_ref_t target_ref = iree_hal_make_indirect_buffer_ref(
      target_buffer_slot, target_offset, length);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r3, &source_ref.buffer));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r5, &target_ref.buffer));
  return iree_hal_command_buffer_copy_buffer(command_buffer, source_ref,
                                             target_ref, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_collective,  //
                   iree_hal_module_state_t,                    //
                   rriiiirrIIIII, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_channel_t* channel = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_channel_check_deref(args->r1, &channel));
  iree_hal_collective_op_t op = {.packed = args->i2};
  uint32_t param = args->i3;
  uint32_t send_buffer_slot = (uint32_t)args->i4;
  uint32_t recv_buffer_slot = (uint32_t)args->i5;
  iree_hal_buffer_ref_t send_ref = iree_hal_make_indirect_buffer_ref(
      send_buffer_slot, iree_hal_cast_device_size(args->i8),
      iree_hal_cast_device_size(args->i9));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r6, &send_ref.buffer));
  iree_hal_buffer_ref_t recv_ref = iree_hal_make_indirect_buffer_ref(
      recv_buffer_slot, iree_hal_cast_device_size(args->i10),
      iree_hal_cast_device_size(args->i11));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref_or_null(args->r7, &recv_ref.buffer));
  iree_device_size_t element_count = iree_hal_cast_device_size(args->i12);
  return iree_hal_command_buffer_collective(command_buffer, channel, op, param,
                                            send_ref, recv_ref, element_count);
}

// Argument signature: rriiiiICiDCiirIID
typedef struct {
  union {
    struct {
      iree_vm_ref_t command_buffer;
      iree_vm_ref_t executable;
      int32_t entry_point;
      uint32_t workgroup_count[3];
      iree_hal_dispatch_flags_t flags;
    };
    iree_vm_abi_rriiiiI_t params;
  };
  iree_vm_size_t constant_count;
  const uint32_t* constants;
  iree_vm_size_t binding_count;
  const iree_vm_abi_iirII_t* bindings;
} iree_hal_module_command_buffer_dispatch_args_t;
static iree_status_t iree_hal_module_command_buffer_dispatch(
    iree_vm_stack_t* IREE_RESTRICT stack, void* IREE_RESTRICT module,
    iree_hal_module_state_t* IREE_RESTRICT state,
    const iree_hal_module_command_buffer_dispatch_args_t* IREE_RESTRICT args) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_check_deref(args->command_buffer,
                                                           &command_buffer));
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_check_deref(args->executable, &executable));

  if (IREE_UNLIKELY(args->binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "binding count %" PRIhsz " > %" PRIhsz,
                            (iree_host_size_t)args->binding_count,
                            IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_buffer_ref_list_t bindings = {
      .count = (iree_host_size_t)args->binding_count,
      .values = (iree_hal_buffer_ref_t*)iree_alloca(
          args->binding_count * sizeof(iree_hal_buffer_ref_t)),
  };
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    iree_hal_buffer_ref_t* binding =
        (iree_hal_buffer_ref_t*)&bindings.values[i];
    binding->reserved = 0;
    binding->buffer_slot = (uint32_t)args->bindings[i].i1;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref_or_null(
        args->bindings[i].r2, &binding->buffer));
    binding->offset = iree_hal_cast_device_size(args->bindings[i].i3);
    binding->length = iree_hal_cast_device_size(args->bindings[i].i4);
  }

  return iree_hal_command_buffer_dispatch(
      command_buffer, executable, args->entry_point, args->workgroup_count,
      iree_make_const_byte_span(args->constants,
                                args->constant_count * sizeof(uint32_t)),
      bindings, (iree_hal_dispatch_flags_t)args->flags);
}
static iree_status_t iree_hal_module_command_buffer_dispatch_shim(
    iree_vm_stack_t* IREE_RESTRICT stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state) {
  // TODO(benvanik): support multiple variadic segments in one call.
  // For now we inline what it would do in a very painful way.
  bool args_ok = true;
  if (args_storage.data_length <
      (sizeof(iree_vm_abi_rriiiiI_t) + sizeof(iree_vm_size_t) +
       sizeof(iree_vm_size_t))) {
    // Can't fit even with zero lengths.
    args_ok = false;
  }
  iree_hal_module_command_buffer_dispatch_args_t args = {
      .params = *(const iree_vm_abi_rriiiiI_t*)args_storage.data,
  };
  if (args_ok) {
    const uint8_t* constants_ptr = args_storage.data + sizeof(args.params);
    args.constant_count = *(const iree_vm_size_t*)constants_ptr;
    args.constants = (const uint32_t*)(constants_ptr + sizeof(iree_vm_size_t));
    const uint8_t* bindings_ptr =
        constants_ptr + sizeof(iree_vm_size_t) +
        args.constant_count * sizeof(args.constants[0]);
    args.binding_count = *(const iree_vm_size_t*)bindings_ptr;
    args.bindings =
        (const iree_vm_abi_iirII_t*)(bindings_ptr + sizeof(iree_vm_size_t));
    const uint8_t* max_ptr = (const uint8_t*)args.bindings +
                             args.binding_count * sizeof(args.bindings[0]);
    const uint8_t* end_ptr = args_storage.data + args_storage.data_length;
    if (max_ptr > end_ptr) args_ok = false;
  }
  if (IREE_UNLIKELY(!args_ok || rets_storage.data_length > 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }
  IREE_ASSERT(target_fn == (iree_vm_native_function_target2_t)
                               iree_hal_module_command_buffer_dispatch);
  return iree_hal_module_command_buffer_dispatch(stack, module, module_state,
                                                 &args);
}

// Argument signature: rriirIICiDCiirIID
typedef struct {
  union {
    struct {
      iree_vm_ref_t command_buffer;
      iree_vm_ref_t executable;
      int32_t entry_point;
      int32_t workgroups_buffer_slot;
      iree_vm_ref_t workgroups_buffer;
      int64_t workgroups_offset;
      iree_hal_dispatch_flags_t flags;
    };
    iree_vm_abi_rriirII_t params;
  };
  iree_vm_size_t constant_count;
  const uint32_t* constants;
  iree_vm_size_t binding_count;
  const iree_vm_abi_iirII_t* bindings;
} iree_hal_module_command_buffer_dispatch_indirect_args_t;
static iree_status_t iree_hal_module_command_buffer_dispatch_indirect(
    iree_vm_stack_t* IREE_RESTRICT stack, void* IREE_RESTRICT module,
    iree_hal_module_state_t* IREE_RESTRICT state,
    const iree_hal_module_command_buffer_dispatch_indirect_args_t* IREE_RESTRICT
        args) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_check_deref(args->command_buffer,
                                                           &command_buffer));
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_check_deref(args->executable, &executable));
  iree_hal_buffer_ref_t workgroups_ref = iree_hal_make_indirect_buffer_ref(
      args->workgroups_buffer_slot,
      iree_hal_cast_device_size(args->workgroups_offset), 3 * sizeof(uint32_t));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref_or_null(
      args->workgroups_buffer, &workgroups_ref.buffer));

  if (IREE_UNLIKELY(args->binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "binding count %" PRIhsz " > %" PRIhsz,
                            (iree_host_size_t)args->binding_count,
                            IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_buffer_ref_list_t bindings = {
      .count = (iree_host_size_t)args->binding_count,
      .values = (iree_hal_buffer_ref_t*)iree_alloca(
          args->binding_count * sizeof(iree_hal_buffer_ref_t)),
  };
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    iree_hal_buffer_ref_t* binding =
        (iree_hal_buffer_ref_t*)&bindings.values[i];
    binding->buffer_slot = (uint32_t)args->bindings[i].i1;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref_or_null(
        args->bindings[i].r2, &binding->buffer));
    binding->offset = iree_hal_cast_device_size(args->bindings[i].i3);
    binding->length = iree_hal_cast_device_size(args->bindings[i].i4);
  }

  return iree_hal_command_buffer_dispatch_indirect(
      command_buffer, executable, args->entry_point, workgroups_ref,
      iree_make_const_byte_span(args->constants,
                                args->constant_count * sizeof(uint32_t)),
      bindings, (iree_hal_dispatch_flags_t)args->flags);
}
static iree_status_t iree_hal_module_command_buffer_dispatch_indirect_shim(
    iree_vm_stack_t* IREE_RESTRICT stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state) {
  // TODO(benvanik): support multiple variadic segments in one call.
  // For now we inline what it would do in a very painful way.
  bool args_ok = true;
  if (args_storage.data_length <
      (sizeof(iree_vm_abi_rriirII_t) + sizeof(iree_vm_size_t) +
       sizeof(iree_vm_size_t))) {
    // Can't fit even with zero lengths.
    args_ok = false;
  }
  iree_hal_module_command_buffer_dispatch_indirect_args_t args = {
      .params = *(const iree_vm_abi_rriirII_t*)args_storage.data,
  };
  if (args_ok) {
    const uint8_t* constants_ptr = args_storage.data + sizeof(args.params);
    args.constant_count = *(const iree_vm_size_t*)constants_ptr;
    args.constants = (const uint32_t*)(constants_ptr + sizeof(iree_vm_size_t));
    const uint8_t* bindings_ptr =
        constants_ptr + sizeof(iree_vm_size_t) +
        args.constant_count * sizeof(args.constants[0]);
    args.binding_count = *(const iree_vm_size_t*)bindings_ptr;
    args.bindings =
        (const iree_vm_abi_iirII_t*)(bindings_ptr + sizeof(iree_vm_size_t));
    const uint8_t* max_ptr = (const uint8_t*)args.bindings +
                             args.binding_count * sizeof(args.bindings[0]);
    const uint8_t* end_ptr = args_storage.data + args_storage.data_length;
    if (max_ptr > end_ptr) args_ok = false;
  }
  if (IREE_UNLIKELY(!args_ok || rets_storage.data_length > 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }
  IREE_ASSERT(target_fn ==
              (iree_vm_native_function_target2_t)
                  iree_hal_module_command_buffer_dispatch_indirect);
  return iree_hal_module_command_buffer_dispatch_indirect(stack, module,
                                                          module_state, &args);
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

IREE_VM_ABI_EXPORT(iree_hal_module_device_query_i64,  //
                   iree_hal_module_state_t,           //
                   rrr, iI) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_vm_buffer_t* category = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &category));
  iree_string_view_t category_str = iree_vm_buffer_as_string(category);
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r2, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  int64_t value = 0;
  iree_status_t query_status =
      iree_hal_device_query_i64(device, category_str, key_str, &value);
  rets->i0 = iree_status_consume_code(query_status) == IREE_STATUS_OK ? 1 : 0;
  rets->i1 = value;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_alloca,  //
                   iree_hal_module_state_t,              //
                   rIrrIiiII, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_allocator_pool_t pool = (iree_hal_allocator_pool_t)args->i4;
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i5;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i6;
  iree_device_size_t allocation_size = iree_hal_cast_device_size(args->i7);
  iree_hal_alloca_flags_t flags = (iree_hal_alloca_flags_t)args->i8;

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), pool, params,
      allocation_size, flags, &buffer));

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_dealloca,  //
                   iree_hal_module_state_t,                //
                   rIrrrI, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r4, &buffer));
  iree_hal_dealloca_flags_t flags = (iree_hal_dealloca_flags_t)args->i5;
  return iree_hal_device_queue_dealloca(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), buffer, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_fill,  //
                   iree_hal_module_state_t,            //
                   rIrrrIIIiI, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r4, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i5);
  iree_device_size_t length = iree_hal_cast_device_size(args->i6);
  uint64_t pattern = args->i7;
  iree_host_size_t pattern_length = iree_hal_cast_host_size(args->i8);
  iree_hal_fill_flags_t flags = (iree_hal_fill_flags_t)args->i9;
  return iree_hal_device_queue_fill(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), target_buffer, target_offset,
      length, &pattern, pattern_length, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_update,  //
                   iree_hal_module_state_t,              //
                   rIrrrIrIII, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_vm_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r4, &source_buffer));
  iree_host_size_t source_offset = iree_hal_cast_host_size(args->i5);
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r6, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i7);
  iree_device_size_t length = iree_hal_cast_device_size(args->i8);
  iree_hal_update_flags_t flags = (iree_hal_update_flags_t)args->i9;
  iree_const_byte_span_t source_span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(source_buffer, source_offset,
                                             length, 1, &source_span));
  return iree_hal_device_queue_update(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), source_span.data, 0,
      target_buffer, target_offset, length, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_copy,  //
                   iree_hal_module_state_t,            //
                   rIrrrIrIII, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r4, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i5);
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r6, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i7);
  iree_device_size_t length = iree_hal_cast_device_size(args->i8);
  iree_hal_copy_flags_t flags = (iree_hal_copy_flags_t)args->i9;
  return iree_hal_device_queue_copy(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), source_buffer, source_offset,
      target_buffer, target_offset, length, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_read,  //
                   iree_hal_module_state_t,            //
                   rIrrrIrIII, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_file_t* source_file = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_file_check_deref(args->r4, &source_file));
  uint64_t source_offset = (uint64_t)args->i5;
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r6, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i7);
  iree_device_size_t length = iree_hal_cast_device_size(args->i8);
  iree_hal_read_flags_t flags = (iree_hal_read_flags_t)args->i9;
  return iree_hal_device_queue_read(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), source_file, source_offset,
      target_buffer, target_offset, length, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_write,  //
                   iree_hal_module_state_t,             //
                   rIrrrIrIII, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r4, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i5);
  iree_hal_file_t* target_file = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_file_check_deref(args->r6, &target_file));
  uint64_t target_offset = (uint64_t)args->i7;
  iree_device_size_t length = iree_hal_cast_device_size(args->i8);
  iree_hal_write_flags_t flags = (iree_hal_write_flags_t)args->i9;
  return iree_hal_device_queue_write(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), source_buffer, source_offset,
      target_file, target_offset, length, flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_barrier,  //
                   iree_hal_module_state_t,               //
                   rIrrI, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_execute_flags_t flags = (iree_hal_execute_flags_t)args->i4;
  return iree_hal_device_queue_barrier(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_execute,  //
                   iree_hal_module_state_t,               //
                   rIrrrI, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r4, &command_buffer));
  iree_hal_execute_flags_t flags = (iree_hal_execute_flags_t)args->i5;
  return iree_hal_device_queue_execute(
      device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), command_buffer,
      iree_hal_buffer_binding_table_empty(), flags);
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_execute_indirect,  //
                   iree_hal_module_state_t,                        //
                   rIrrrICrIID, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r4, &command_buffer));
  iree_hal_execute_flags_t flags = (iree_hal_execute_flags_t)args->i5;

  // Allocate temporary storage for the binding table in order to marshal VM
  // refs and 64-bit offsets/lengths into the types required by the HAL C API.
  iree_host_size_t binding_count = args->a6_count;
  iree_hal_buffer_binding_t* bindings = NULL;
  if (binding_count > IREE_HAL_MODULE_MAX_STACK_COMMAND_BUFFER_BINDING_COUNT) {
    // Heap allocate when using a large number of bindings to avoid blowing the
    // native stack. Note that we have to free it before returning from the
    // function.
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        state->host_allocator, binding_count * sizeof(*bindings),
        (void**)&bindings));
  } else {
    // Stack allocate when using a small number of bindings (common).
    bindings = (iree_hal_buffer_binding_t*)iree_alloca(binding_count *
                                                       sizeof(*bindings));
  }

  // Ensure all buffers are valid (may be NULL) and build the binding table.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    status = iree_hal_buffer_check_deref_or_null(args->a6[i].r0,
                                                 &bindings[i].buffer);
    if (!iree_status_is_ok(status)) break;
    bindings[i].offset = iree_hal_cast_device_size(args->a6[i].i1);
    bindings[i].length = iree_hal_cast_device_size(args->a6[i].i2);
  }

  // Schedule execution with the binding table - it will be copied by the device
  // and need not live longer than the call.
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t binding_table = {
        .count = binding_count,
        .bindings = bindings,
    };
    status = iree_hal_device_queue_execute(
        device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), command_buffer,
        binding_table, flags);
  }

  // If we had to heap-allocate the binding table storage it must be freed
  // before returning to the VM.
  if (binding_count > IREE_HAL_MODULE_MAX_STACK_COMMAND_BUFFER_BINDING_COUNT) {
    iree_allocator_free(state->host_allocator, bindings);
  }

  return status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_flush,  //
                   iree_hal_module_state_t,             //
                   rI, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  return iree_hal_device_queue_flush(device, queue_affinity);
}

//===----------------------------------------------------------------------===//
// iree_hal_device_t management
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_devices_count,  //
                   iree_hal_module_state_t,        //
                   v, i) {
  rets->i0 = (int32_t)state->device_count;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_devices_get,  //
                   iree_hal_module_state_t,      //
                   i, r) {
  if (args->i0 < state->device_count) {
    rets->r0 = iree_hal_device_retain_ref(state->devices[args->i0]);
  } else {
    rets->r0 = iree_vm_ref_null();
  }
  return iree_ok_status();
}

//===--------------------------------------------------------------------===//
// iree_hal_executable_t
//===--------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_executable_create,  //
                   iree_hal_module_state_t,            //
                   rIrrr, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_vm_buffer_t* executable_format = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref(args->r2, &executable_format));
  iree_string_view_t executable_format_str =
      iree_vm_buffer_as_string(executable_format);
  iree_vm_buffer_t* executable_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r3, &executable_data));
  iree_host_size_t constant_count = 0;
  const uint32_t* constants = NULL;
  if (iree_vm_buffer_isa(args->r4)) {
    iree_vm_buffer_t* constant_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_check_deref(args->r4, &constant_buffer));
    if (constant_buffer->data.data_length % 4 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant buffer data must contain 4-byte "
                              "elements but data length is %" PRIhsz,
                              constant_buffer->data.data_length);
    }
    constant_count = constant_buffer->data.data_length / sizeof(uint32_t);
    constants = (const uint32_t*)constant_buffer->data.data;
  }

  iree_hal_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_module_state_lookup_executable_cache(
      state, device, &executable_cache));

  iree_hal_executable_t* executable = NULL;
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.queue_affinity = queue_affinity;
  executable_params.caching_mode |=
      executable_data->access == IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE
          ? IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA
          : 0;
  executable_params.executable_format = executable_format_str;
  executable_params.executable_data = iree_make_const_byte_span(
      executable_data->data.data, executable_data->data.data_length);
  executable_params.constant_count = constant_count;
  executable_params.constants = constants;
  IREE_RETURN_IF_ERROR(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  rets->r0 = iree_hal_executable_move_ref(executable);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_fence_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_fence_create,  //
                   iree_hal_module_state_t,       //
                   rI, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  uint64_t fence_flags = args->i1;
  (void)fence_flags;

  // TODO(benvanik): hide semaphores from the API.
  // This should be reworked to just create the fence.

  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Create fence with room for our single semaphore.
  iree_hal_fence_t* fence = NULL;
  iree_status_t status =
      iree_hal_fence_create(1, state->host_allocator, &fence);
  if (iree_status_is_ok(status)) {
    status = iree_hal_fence_insert(fence, semaphore, 1ull);
  }

  iree_hal_semaphore_release(semaphore);
  if (iree_status_is_ok(status)) {
    rets->r0 = iree_hal_fence_move_ref(fence);
  } else {
    iree_hal_fence_release(fence);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_join,  //
                   iree_hal_module_state_t,     //
                   ICrD, r) {
  // NOTE: this is an inlined version of iree_hal_fence_join that avoids the
  // need for mapping VM types to HAL types via temporary stack/heap storage.
  // This lets us avoid allocations/stack exhaustion in pathological cases of
  // hundreds of fences (say, one per input argument in stateless programs with
  // hundreds/thousands of inputs).

  uint64_t fence_flags = args->i0;
  (void)fence_flags;

  // Find the maximum required timepoint capacity by scanning the fence list.
  // This ensures all fences passed in are actually fences _or_ are NULL so
  // the subsequent scan below only needs to check for NULL cases.
  iree_host_size_t total_timepoint_capacity = 0;
  for (iree_host_size_t i = 0; i < args->a1_count; ++i) {
    iree_hal_fence_t* fence = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_fence_check_deref_or_null(args->a1[i].r0, &fence));
    if (fence) {
      total_timepoint_capacity += iree_hal_fence_timepoint_count(fence);
    }
  }

  // If all fences were empty then we no-op by returning a NULL fence
  // (immediately signaled).
  if (!total_timepoint_capacity) {
    rets->r0 = iree_vm_ref_null();
    return iree_ok_status();
  }

  // Create the fence with the maximum capacity. Hopefully there is some
  // deduplication.
  iree_hal_fence_t* joined_fence = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_fence_create(
      total_timepoint_capacity, state->host_allocator, &joined_fence));

  // Insert all timepoints from all fences. This is slow in cases where there
  // are a lot of unique fences.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < args->a1_count; ++i) {
    // NOTE: only possible because we checked above and know this is NULL or an
    // iree_hal_fence_t.
    iree_hal_fence_t* fence = (iree_hal_fence_t*)args->a1[i].r0.ptr;
    if (!fence) continue;
    iree_hal_semaphore_list_t source_list =
        iree_hal_fence_semaphore_list(fence);
    for (iree_host_size_t j = 0; j < source_list.count; ++j) {
      status = iree_hal_fence_insert(joined_fence, source_list.semaphores[j],
                                     source_list.payload_values[j]);
      if (!iree_status_is_ok(status)) break;
    }
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    rets->r0 = iree_hal_fence_move_ref(joined_fence);
  } else {
    iree_hal_fence_release(joined_fence);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_query,  //
                   iree_hal_module_state_t,      //
                   r, i) {
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_fence_check_deref(args->r0, &fence));

  iree_status_t query_status = iree_hal_fence_query(fence);
  rets->i0 = iree_status_consume_code(query_status);
  iree_status_ignore(query_status);

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_signal,  //
                   iree_hal_module_state_t,       //
                   r, v) {
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_fence_check_deref(args->r0, &fence));
  return iree_hal_fence_signal(fence);
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_fail,  //
                   iree_hal_module_state_t,     //
                   ri, v) {
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_fence_check_deref(args->r0, &fence));
  iree_status_code_t status_code =
      (iree_status_code_t)(args->i1 & IREE_STATUS_CODE_MASK);
  iree_hal_fence_fail(fence, iree_make_status(status_code));
  return iree_ok_status();
}

// Removes entries in |fences| if they have been reached.
// Returns failure if one or more fences have failed.
static iree_status_t iree_hal_module_fence_elide_reached(
    iree_host_size_t* fence_count, iree_hal_fence_t** fences) {
  iree_host_size_t new_count = *fence_count;
  for (iree_host_size_t i = 0; i < new_count;) {
    iree_status_t status = iree_hal_fence_query(fences[i]);
    if (iree_status_is_ok(status)) {
      // Has been reached; shift the list down.
      memmove(&fences[i], &fences[i + 1],
              (new_count - i - 1) * sizeof(iree_hal_fence_t*));
      fences[new_count - 1] = NULL;
      --new_count;
    } else if (iree_status_is_deferred(status)) {
      // Still waiting.
      iree_status_ignore(status);
      ++i;  // next
    } else {
      // Failed; propagate failure.
      *fence_count = new_count;
      return status;
    }
  }
  *fence_count = new_count;
  return iree_ok_status();
}

// Enters a wait frame for all timepoints in all |fences|.
// Returns an |out_wait_status| of OK if all fences have been reached or
// IREE_STATUS_DEFERRED if one or more fences are still pending and a wait
// frame was entered.
static iree_status_t iree_hal_module_fence_await_begin(
    iree_vm_stack_t* stack, iree_host_size_t fence_count,
    iree_hal_fence_t** fences, iree_timeout_t timeout, iree_zone_id_t zone_id,
    iree_status_t* out_wait_status) {
  // To avoid additional allocations when waiting on multiple fences we enter
  // the wait frame with the maximum required wait source capacity and perform
  // a simple deduplication when building the list. Ideally this helps get us on
  // fast paths of single semaphore waits. The common case is a single fence in
  // which case this is all exceptional.
  iree_host_size_t total_timepoint_capacity = 0;
  for (iree_host_size_t i = 0; i < fence_count; ++i) {
    total_timepoint_capacity += iree_hal_fence_timepoint_count(fences[i]);
  }

  // Fast-path for no semaphores (empty/immediate fences).
  if (total_timepoint_capacity == 0) {
    *out_wait_status = iree_ok_status();
    IREE_TRACE_ZONE_END(zone_id);
    return iree_ok_status();
  }

  // Reserve storage as if all timepoints from all fences were unique.
  iree_vm_wait_frame_t* wait_frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_stack_wait_enter(stack, IREE_VM_WAIT_ALL,
                                                total_timepoint_capacity,
                                                timeout, zone_id, &wait_frame));

  // Insert the first set of timepoints - they're already deduplicated.
  iree_host_size_t unique_timepoint_count = 0;
  if (fence_count >= 1) {
    iree_hal_semaphore_list_t semaphore_list =
        iree_hal_fence_semaphore_list(fences[0]);
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      iree_wait_source_t wait_source = iree_hal_semaphore_await(
          semaphore_list.semaphores[i], semaphore_list.payload_values[i]);
      wait_frame->wait_sources[unique_timepoint_count++] = wait_source;
    }
  }

  // TODO(benvanik): simplify this; it may not be worth the complexity. We'll
  // need more real workloads using multi-fence joins to see how useful this is.

  // Insert remaining fence timepoints by performing merging as we go.
  for (iree_host_size_t i = 1; i < fence_count; ++i) {
    iree_hal_semaphore_list_t semaphore_list =
        iree_hal_fence_semaphore_list(fences[i]);
    for (iree_host_size_t j = 0; j < semaphore_list.count; ++j) {
      // O(n^2) set insertion - relying on this being rare and the total count
      // being low. The savings of a small linear scan here relative to an
      // additional syscall are always worth it but we may want to go further.
      iree_wait_source_t wait_source = iree_hal_semaphore_await(
          semaphore_list.semaphores[j], semaphore_list.payload_values[j]);
      bool found_existing = false;
      for (iree_host_size_t k = 0; k < unique_timepoint_count; ++k) {
        if (wait_frame->wait_sources[k].ctl == wait_source.ctl &&
            wait_frame->wait_sources[k].self == wait_source.self) {
          // Found existing; use max of both.
          wait_frame->wait_sources[k].data =
              iree_max(wait_frame->wait_sources[k].data, wait_source.data);
          found_existing = true;
          break;
        }
      }
      if (!found_existing) {
        wait_frame->wait_sources[unique_timepoint_count++] = wait_source;
      }
    }
  }

  // Update frame with the actual number of timepoints in the wait operation.
  wait_frame->count = unique_timepoint_count;

  *out_wait_status = iree_status_from_code(IREE_STATUS_DEFERRED);
  return iree_ok_status();
}

// PC for iree_hal_module_fence_await.
enum iree_hal_module_fence_await_pc_e {
  // Initial entry point that will try to either wait inline or yield to the
  // scheduler with a wait-all operation.
  IREE_HAL_MODULE_FENCE_AWAIT_PC_BEGIN = 0,
  // Resume entry point after the scheduler wait has resolved (successfully or
  // otherwise).
  IREE_HAL_MODULE_FENCE_AWAIT_PC_RESUME,
};

IREE_VM_ABI_EXPORT(iree_hal_module_fence_await,  //
                   iree_hal_module_state_t,      //
                   iICrD, i) {
  // On entry we either perform the wait or begin a coroutine yield operation.
  // After resuming we check to see if the fence has been reached and propagate
  // the result.
  iree_vm_stack_frame_t* current_frame = iree_vm_stack_top(stack);
  iree_zone_id_t zone_id = 0;
  iree_status_t wait_status = iree_ok_status();
  if (current_frame->pc == IREE_HAL_MODULE_FENCE_AWAIT_PC_BEGIN) {
    uint32_t timeout_millis = (uint32_t)args->i0;
    uint64_t flags = (uint64_t)args->i1;
    (void)flags;  // unused today
    iree_host_size_t fence_count = 0;
    iree_hal_fence_t** fences = NULL;
    IREE_VM_ABI_VLA_STACK_DEREF_OR_NULL(args, a2_count, a2, iree_hal_fence, 32,
                                        &fence_count, &fences);

    IREE_TRACE_ZONE_BEGIN(z0);
    zone_id = z0;

    // Capture absolute timeout so that regardless of how long it takes us to
    // wait the user-perceived wait time remains the same.
    iree_timeout_t timeout = timeout_millis == UINT32_MAX
                                 ? iree_infinite_timeout()
                                 : iree_make_timeout_ms(timeout_millis);
    iree_convert_timeout_to_absolute(&timeout);

    // Remove any fences that have been reached and check for failure.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        zone_id, iree_hal_module_fence_elide_reached(&fence_count, fences));

    // If all fences have been reached we can exit early as if we waited
    // successfully.
    if (fence_count > 0) {
      if (iree_all_bits_set(state->flags, IREE_HAL_MODULE_FLAG_SYNCHRONOUS)) {
        // Block the native thread until the fence is reached or the deadline is
        // exceeded.
        for (iree_host_size_t i = 0; i < fence_count; ++i) {
          wait_status = iree_hal_fence_wait(fences[i], timeout);
          if (!iree_status_is_ok(wait_status)) break;
        }
      } else {
        current_frame->pc = IREE_HAL_MODULE_FENCE_AWAIT_PC_RESUME;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            zone_id,
            iree_hal_module_fence_await_begin(stack, fence_count, fences,
                                              timeout, zone_id, &wait_status));
        if (iree_status_is_deferred(wait_status)) {
          zone_id = 0;  // ownership transferred to wait frame
        }
      }
    }
  } else {
    // Resume by leaving the wait frame and storing the result.
    iree_vm_wait_result_t wait_result;
    IREE_RETURN_IF_ERROR(iree_vm_stack_wait_leave(stack, &wait_result));
    wait_status = wait_result.status;
    IREE_TRACE(zone_id = wait_result.trace_zone);
  }

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(wait_status)) {
    // Successful wait.
    rets->i0 = 0;
  } else if (iree_status_is_deferred(wait_status)) {
    // Yielding; resume required.
    // NOTE: zone not ended as it's reserved on the stack.
    status = wait_status;
  } else if (iree_status_is_deadline_exceeded(wait_status)) {
    // Propagate deadline exceeded back to the VM.
    rets->i0 = (int32_t)iree_status_consume_code(wait_status);
    iree_status_ignore(wait_status);
  } else {
    // Fail the invocation.
    status = wait_status;
  }

  IREE_TRACE({
    if (zone_id) IREE_TRACE_ZONE_END(zone_id);
  });
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
#define EXPORT_FN_CUSTOM(name, target_fn, arg_types, ret_types)   \
  {                                                               \
      .shim = (iree_vm_native_function_shim_t)(target_fn##_shim), \
      .target = (iree_vm_native_function_target_t)(target_fn),    \
  },
#include "iree/modules/hal/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
#undef EXPORT_FN_CUSTOM
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_hal_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_hal_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#define EXPORT_FN_CUSTOM EXPORT_FN
#include "iree/modules/hal/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
#undef EXPORT_FN_CUSTOM
};
static_assert(IREE_ARRAYSIZE(iree_hal_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_hal_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_hal_module_descriptor_ = {
    .name = iree_string_view_literal("hal"),
    .version = IREE_HAL_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0,  // workaround for 0-length C struct
    .imports = iree_hal_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_hal_module_exports_),
    .exports = iree_hal_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_hal_module_funcs_),
    .functions = iree_hal_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_hal_module_create(
    iree_vm_instance_t* instance, iree_host_size_t device_count,
    iree_hal_device_t** devices, iree_hal_module_flags_t flags,
    iree_hal_module_debug_sink_t debug_sink, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(device_count);
  IREE_ASSERT_ARGUMENT(devices);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_hal_module_destroy,
      .alloc_state = iree_hal_module_alloc_state,
      .free_state = iree_hal_module_free_state,
      .fork_state = iree_hal_module_fork_state,
      .notify = iree_hal_module_notify,
  };

  // Allocate shared module state.
  iree_host_size_t total_size = iree_vm_native_module_size() +
                                sizeof(iree_hal_module_t) +
                                device_count * sizeof(iree_hal_device_t*);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status =
      iree_vm_native_module_initialize(&interface, &iree_hal_module_descriptor_,
                                       instance, host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  // TODO(benvanik): fix vm yield with result storage.
  module->flags = flags | IREE_HAL_MODULE_FLAG_SYNCHRONOUS;
  module->debug_sink = debug_sink;
  module->device_count = device_count;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    module->devices[i] = devices[i];
    iree_hal_device_retain(module->devices[i]);
  }

  *out_module = base_module;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_host_size_t
iree_hal_module_state_device_count(iree_vm_module_state_t* module_state) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  return state->device_count;
}

IREE_API_EXPORT iree_hal_device_t* iree_hal_module_state_device_get(
    iree_vm_module_state_t* module_state, iree_host_size_t index) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  return index < state->device_count ? state->devices[index] : NULL;
}
