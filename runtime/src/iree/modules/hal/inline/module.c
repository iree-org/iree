// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/inline/module.h"

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/utils/buffer_diagnostics.h"
#include "iree/vm/api.h"

#define IREE_HAL_INLINE_MODULE_VERSION_0_0 0x00000000u
#define IREE_HAL_INLINE_MODULE_VERSION_LATEST IREE_HAL_INLINE_MODULE_VERSION_0_0

//===----------------------------------------------------------------------===//
// iree_hal_inline_storage_buffer_t
//===----------------------------------------------------------------------===//

// Inlined VM buffer using a HAL buffer for storage.
// This uses the reference counting of the embedded VM buffer
// to track lifetime combined with a custom allocator to handle
// cleaning up this wrapper when the VM buffer is no longer referenced.
//
// Since the HAL buffer is providing the storage and the VM buffer is just
// pointing into it the critical thing this wrapper does is ensure the HAL
// buffer always outlives the VM buffer.
//
// NOTE: this is allocated each storage query! The assumption is that the
// returned buffer is long-lived (at least per-invocation). This is primarily
// used to get the backing storage of a !hal.buffer that a user passes into an
// invocation and the compiler should CSE such queries. Since users can provide
// their own allocators they can decide if they want to pool small allocations
// to bypass the system allocator. If we wanted to in here we could have a small
// free list we maintained for this purpose at the cost of fixed memory
// consumption. Note that the key requirement is that the returned VM buffer
// may outlive the module so we can't use an arena that has module lifetime.
typedef struct iree_hal_inline_storage_buffer_t {
  // Allocator used to allocate this storage buffer.
  iree_allocator_t host_allocator;
  // HAL buffer backing this storage buffer.
  // Retained for the lifetime of this instance so that the
  // wrapped vm_buffer is always valid.
  iree_hal_buffer_t* hal_buffer;
  // Scoped mapping into the buffer. We could make it persistent but because
  // we can trivially scope things having this extra information is cheap and
  // useful for debugging.
  iree_hal_buffer_mapping_t mapping;
  // Inline initialized VM buffer wrapping the hal_buffer storage.
  // This directly references the memory of the HAL buffer.
  // The buffer has a custom allocator that calls back into this
  // struct to deallocate the wrapper.
  iree_vm_buffer_t vm_buffer;
} iree_hal_inline_storage_buffer_t;

static void iree_hal_inline_storage_buffer_destroy(
    iree_hal_inline_storage_buffer_t* storage);

static iree_status_t iree_hal_inline_storage_buffer_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  if (command != IREE_ALLOCATOR_COMMAND_FREE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocator can only be used for dropping the wrapper buffer");
  }
  iree_hal_inline_storage_buffer_t* storage =
      (iree_hal_inline_storage_buffer_t*)self;
  iree_hal_inline_storage_buffer_destroy(storage);
  return iree_ok_status();
}

// Creates a VM buffer wrapper that directly references HAL buffer storage.
// The returned |out_vm_buffer| lifetime will extend the HAL buffer lifetime.
static iree_status_t iree_hal_inline_storage_buffer_create(
    iree_hal_buffer_t* hal_buffer, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_vm_buffer) {
  IREE_ASSERT_ARGUMENT(hal_buffer);
  IREE_ASSERT_ARGUMENT(out_vm_buffer);
  *out_vm_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate zero-initialized storage wrapper.
  iree_hal_inline_storage_buffer_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*storage),
                                (void**)&storage));

  // Map the HAL buffer into host-accessible memory. It almost always is but
  // it's possible the buffer we were passed was allocated on a real device that
  // requires mapping.
  iree_status_t status = iree_hal_buffer_map_range(
      hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ANY, 0,
      IREE_WHOLE_BUFFER, &storage->mapping);

  // Initializes the VM buffer to reference the mapped memory.
  // Since the VM buffer is what we pass back to the VM and gets reference
  // counted we pass a custom allocator that lets us know when the VM (or
  // user) is done with it.
  if (iree_status_is_ok(status)) {
    iree_allocator_t self_allocator = {
        .self = storage,
        .ctl = iree_hal_inline_storage_buffer_ctl,
    };
    iree_vm_buffer_initialize(
        IREE_VM_BUFFER_ACCESS_ORIGIN_HOST | IREE_VM_BUFFER_ACCESS_MUTABLE,
        storage->mapping.contents, self_allocator, &storage->vm_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_vm_buffer = &storage->vm_buffer;
  } else {
    iree_hal_inline_storage_buffer_destroy(storage);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_inline_storage_buffer_destroy(
    iree_hal_inline_storage_buffer_t* storage) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = storage->host_allocator;
  iree_hal_buffer_unmap_range(&storage->mapping);
  iree_hal_buffer_release(storage->hal_buffer);
  iree_allocator_free(host_allocator, storage);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_hal_inline_module_t {
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  iree_hal_inline_module_flags_t flags;
  // TODO(benvanik): types.
} iree_hal_inline_module_t;

#define IREE_HAL_INLINE_MODULE_CAST(module)        \
  (iree_hal_inline_module_t*)((uint8_t*)(module) + \
                              iree_vm_native_module_size());

typedef struct iree_hal_inline_module_state_t {
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  iree_hal_inline_module_flags_t flags;
} iree_hal_inline_module_state_t;

static void IREE_API_PTR iree_hal_inline_module_destroy(void* base_module) {
  iree_hal_inline_module_t* module = IREE_HAL_INLINE_MODULE_CAST(base_module);
  iree_hal_allocator_release(module->device_allocator);
  module->device_allocator = NULL;
}

static iree_status_t IREE_API_PTR
iree_hal_inline_module_alloc_state(void* self, iree_allocator_t host_allocator,
                                   iree_vm_module_state_t** out_module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_inline_module_t* module = IREE_HAL_INLINE_MODULE_CAST(self);
  iree_hal_inline_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->device_allocator = module->device_allocator;
  iree_hal_allocator_retain(state->device_allocator);
  state->flags = module->flags;

  *out_module_state = (iree_vm_module_state_t*)state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR iree_hal_inline_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_inline_module_state_t* state =
      (iree_hal_inline_module_state_t*)module_state;
  iree_hal_allocator_release(state->device_allocator);
  state->device_allocator = NULL;
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_hal_inline_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  switch (signal) {
    case IREE_VM_SIGNAL_SUSPEND:
    case IREE_VM_SIGNAL_LOW_MEMORY:
    default:
      return iree_ok_status();
  }
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Casts a VM value to a HAL device size.
static iree_device_size_t iree_hal_cast_device_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if device
  // size is 32-bits.
  return (iree_device_size_t)value;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_module_buffer_allocate_with_storage(
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer,
    iree_vm_buffer_t** out_storage) {
  // We could optimize this to create both at the same time and avoid the extra
  // storage allocation by having a custom iree_hal_buffer_t type or a way to
  // allocate additional data in the iree_hal_buffer_params_t that we stashed
  // the storage in. Today this is all intentionally simple and something we can
  // change in the runtime without impacting the compiler/artifacts.

  // Allocate the buffer with uninitialized contents.
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator, params, allocation_size, &buffer));

  // Map and retain the HAL buffer and return a VM buffer that is usable as if
  // it were a native iree_vm_buffer_t.
  iree_vm_buffer_t* storage = NULL;
  iree_status_t status =
      iree_hal_inline_storage_buffer_create(buffer, host_allocator, &storage);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }

  // Now that we know we have a host pointer mapped we can copy over the initial
  // data (if any).
  if (!iree_const_byte_span_is_empty(initial_data)) {
    memcpy(iree_vm_buffer_data(storage), initial_data.data,
           iree_min(initial_data.data_length, allocation_size));
  }

  *out_buffer = buffer;
  *out_storage = storage;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_allocate,  //
                   iree_hal_inline_module_state_t,          //
                   iI, rr) {
  iree_device_size_t minimum_alignment = iree_hal_cast_device_size(args->i0);
  iree_device_size_t allocation_size = iree_hal_cast_device_size(args->i1);

  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST,
      .min_alignment = minimum_alignment,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_vm_buffer_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_module_buffer_allocate_with_storage(
      state->device_allocator, params, allocation_size,
      iree_const_byte_span_empty(), state->host_allocator, &buffer, &storage));

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  rets->r1 = iree_vm_buffer_move_ref(storage);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_allocate_initialized,  //
                   iree_hal_inline_module_state_t,                      //
                   irII, rr) {
  iree_device_size_t minimum_alignment = iree_hal_cast_device_size(args->i0);
  iree_vm_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i2);
  iree_device_size_t source_length = iree_hal_cast_device_size(args->i3);

  iree_const_byte_span_t initial_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(source_buffer, source_offset,
                                             source_length, 1, &initial_data));

  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST,
      .min_alignment = minimum_alignment,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_vm_buffer_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_module_buffer_allocate_with_storage(
      state->device_allocator, params, source_length, initial_data,
      state->host_allocator, &buffer, &storage));

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  rets->r1 = iree_vm_buffer_move_ref(storage);
  return iree_ok_status();

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_wrap,  //
                   iree_hal_inline_module_state_t,      //
                   rII, r) {
  iree_vm_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_device_size_t source_length = iree_hal_cast_device_size(args->i2);

  // TODO(benvanik): implement buffer wrapping.
  // We don't emit this on the compiler today but could if we wanted to return
  // constants/variables from the program without copies.
  //
  // We could do this by having a custom iree_hal_buffer_t type that retains
  // the vm buffer, like `iree_hal_external_vm_buffer_t`.
  // We may then want to expose this wrap method on the public module API so
  // that users can pass in buffers like this.
  //
  // hal_inline.buffer.storage would need to switch based on type and return
  // the underlying wrapped vm.buffer.
  (void)source_buffer;
  (void)source_offset;
  (void)source_length;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "vm->hal buffer wrapping not yet implemented");
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_subspan,  //
                   iree_hal_inline_module_state_t,         //
                   rII, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_device_size_t length = iree_hal_cast_device_size(args->i2);

  iree_hal_buffer_t* subspan_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_subspan(source_buffer, source_offset, length,
                              &subspan_buffer),
      "invalid subspan of an existing buffer (source_offset=%" PRIdsz
      ", length=%" PRIdsz ")",
      source_offset, length);

  rets->r0 = iree_hal_buffer_move_ref(subspan_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_length,  //
                   iree_hal_inline_module_state_t,        //
                   r, I) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->i0 = (int64_t)iree_hal_buffer_byte_length(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_storage,  //
                   iree_hal_inline_module_state_t,         //
                   r, r) {
  iree_hal_buffer_t* hal_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &hal_buffer));

  // Map and retain the HAL buffer and return a VM buffer that is usable as if
  // it were a native iree_vm_buffer_t.
  iree_vm_buffer_t* vm_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_storage_buffer_create(
      hal_buffer, state->host_allocator, &vm_buffer));

  rets->r0 = iree_vm_buffer_move_ref(vm_buffer);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_create,  //
                   iree_hal_inline_module_state_t,             //
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
                                &subspan_buffer),
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

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_assert,  //
                   iree_hal_inline_module_state_t,             //
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

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_buffer,  //
                   iree_hal_inline_module_state_t,             //
                   r, r) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->r0 =
      iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_element_type,  //
                   iree_hal_inline_module_state_t,                   //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (uint32_t)iree_hal_buffer_view_element_type(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_encoding_type,  //
                   iree_hal_inline_module_state_t,                    //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (uint32_t)iree_hal_buffer_view_encoding_type(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_rank,  //
                   iree_hal_inline_module_state_t,           //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (iree_vm_size_t)iree_hal_buffer_view_shape_rank(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_dim,  //
                   iree_hal_inline_module_state_t,          //
                   ri, I) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  iree_vm_size_t index = (iree_vm_size_t)args->i1;
  rets->i0 = (int64_t)iree_hal_buffer_view_shape_dim(buffer_view, index);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_inline_module_buffer_view_trace,  //
                   iree_hal_inline_module_state_t,            //
                   rCrD, v) {
  return iree_hal_modules_buffer_view_trace(args->r0, args->a1_count, args->a1,
                                            state->host_allocator);
}

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_inline_module_device_query_i64,  //
                   iree_hal_inline_module_state_t,           //
                   rr, iI) {
  iree_vm_buffer_t* category = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r0, &category));
  iree_string_view_t category_str = iree_vm_buffer_as_string(category);
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  // TODO(benvanik): allow injection of a query function on the module. This
  // would let us extend the queryable configuration with either synthetic
  // properties or user-provided ones. For now we could at least provide
  // compile-time configuration (like hosting architecture) but nothing dynamic
  // (like cache sizes).

  iree_status_t query_status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
  int64_t value = 0;
  if (iree_string_view_equal(category_str, IREE_SV("hal.cpu"))) {
    query_status = iree_cpu_lookup_data_by_key(key_str, &value);
  }

  rets->i0 = iree_status_consume_code(query_status) == IREE_STATUS_OK ? 1 : 0;
  rets->i1 = value;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_hal_inline_module_exports_
// table.
static const iree_vm_native_function_ptr_t iree_hal_inline_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/hal/inline/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t
    iree_hal_inline_module_imports_[1];

static const iree_vm_native_export_descriptor_t
    iree_hal_inline_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#include "iree/modules/hal/inline/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_hal_inline_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_hal_inline_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t
    iree_hal_inline_module_descriptor_ = {
        .name = iree_string_view_literal("hal_inline"),
        .version = IREE_HAL_INLINE_MODULE_VERSION_LATEST,
        .attr_count = 0,
        .attrs = NULL,
        .dependency_count = 0,
        .dependencies = NULL,
        .import_count = 0,  // workaround for 0-length C struct
        .imports = iree_hal_inline_module_imports_,
        .export_count = IREE_ARRAYSIZE(iree_hal_inline_module_exports_),
        .exports = iree_hal_inline_module_exports_,
        .function_count = IREE_ARRAYSIZE(iree_hal_inline_module_funcs_),
        .functions = iree_hal_inline_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_hal_inline_module_create(
    iree_vm_instance_t* instance, iree_hal_inline_module_flags_t flags,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_hal_inline_module_destroy,
      .alloc_state = iree_hal_inline_module_alloc_state,
      .free_state = iree_hal_inline_module_free_state,
      .notify = iree_hal_inline_module_notify,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_hal_inline_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_hal_inline_module_descriptor_, instance, host_allocator,
      base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_hal_inline_module_t* module = IREE_HAL_INLINE_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->device_allocator = device_allocator;
  iree_hal_allocator_retain(module->device_allocator);
  module->flags = flags;

  *out_module = base_module;
  return iree_ok_status();
}
