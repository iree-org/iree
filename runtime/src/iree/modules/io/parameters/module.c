// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/io/parameters/module.h"

#include "iree/modules/hal/types.h"

#define IREE_IO_PARAMETERS_MODULE_VERSION_0_0 0x00000000u
#define IREE_IO_PARAMETERS_MODULE_VERSION_LATEST \
  IREE_IO_PARAMETERS_MODULE_VERSION_0_0

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_io_parameters_module_t {
  iree_allocator_t host_allocator;
  iree_host_size_t provider_count;
  iree_io_parameter_provider_t* providers[];
} iree_io_parameters_module_t;

#define IREE_IO_PARAMETERS_MODULE_CAST(module)        \
  (iree_io_parameters_module_t*)((uint8_t*)(module) + \
                                 iree_vm_native_module_size())

typedef struct iree_io_parameters_module_state_t {
  iree_allocator_t host_allocator;
} iree_io_parameters_module_state_t;

static void IREE_API_PTR iree_io_parameters_module_destroy(void* base_module) {
  iree_io_parameters_module_t* module =
      IREE_IO_PARAMETERS_MODULE_CAST(base_module);
  for (iree_host_size_t i = 0; i < module->provider_count; ++i) {
    iree_io_parameter_provider_release(module->providers[i]);
  }
  module->provider_count = 0;
}

static iree_status_t IREE_API_PTR iree_io_parameters_module_alloc_state(
    void* self, iree_allocator_t host_allocator,
    iree_vm_module_state_t** out_module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameters_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;

  *out_module_state = (iree_vm_module_state_t*)state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR iree_io_parameters_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameters_module_state_t* state =
      (iree_io_parameters_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_io_parameters_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  iree_io_parameters_module_t* module = IREE_IO_PARAMETERS_MODULE_CAST(self);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_io_parameter_provider_signal_t provider_signal;
  switch (signal) {
    case IREE_VM_SIGNAL_RESUME:
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "RESUME");
      provider_signal = IREE_IO_PARAMETER_PROVIDER_SIGNAL_RESUME;
      break;
    case IREE_VM_SIGNAL_SUSPEND:
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "SUSPEND");
      provider_signal = IREE_IO_PARAMETER_PROVIDER_SIGNAL_SUSPEND;
      break;
    case IREE_VM_SIGNAL_LOW_MEMORY:
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "LOW_MEMORY");
      provider_signal = IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY;
      break;
    default:
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(unhandled)");
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < module->provider_count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_provider_notify(module->providers[i],
                                              provider_signal));
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Scans the provider list to find one that handles |scope|.
static iree_status_t iree_io_parameters_module_resolve_provider(
    iree_io_parameters_module_t* module, iree_string_view_t scope,
    iree_io_parameter_provider_t** out_provider) {
  for (iree_host_size_t i = 0; i < module->provider_count; ++i) {
    iree_io_parameter_provider_t* provider = module->providers[i];
    if (iree_io_parameter_provider_query_support(provider, scope)) {
      *out_provider = provider;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "no provider registered that handles scopes like '%.*s'", (int)scope.size,
      scope.data);
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

typedef struct iree_io_parameters_string_entry_t {
  uint32_t offset;
  uint32_t length;
} iree_io_parameters_string_entry_t;

typedef struct iree_io_parameters_span_entry_t {
  uint64_t parameter_offset;
  uint64_t buffer_offset;
  uint64_t length;
} iree_io_parameters_span_entry_t;

typedef struct iree_io_parameters_indirect_args_t {
  iree_host_size_t count;
  const iree_io_parameters_string_entry_t* string_table;
  iree_const_byte_span_t string_data;
  const iree_io_parameters_span_entry_t* spans;
} iree_io_parameters_indirect_args_t;

static iree_status_t iree_io_parameters_prepare_indirect_args(
    iree_vm_buffer_t* key_table, iree_vm_buffer_t* key_data,
    iree_vm_buffer_t* spans, iree_io_parameters_indirect_args_t* out_args) {
  // Span count is defined by the number of entries that storage contains.
  if (iree_vm_buffer_length(spans) % sizeof(iree_io_parameters_span_entry_t) !=
      0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer span storage must be aligned to "
                            "iree_io_parameters_span_entry_t");
  }
  const iree_host_size_t count =
      iree_vm_buffer_length(spans) / sizeof(iree_io_parameters_span_entry_t);

  // Verify there's enough space in the key string table for the entries we
  // need.
  if (iree_vm_buffer_length(key_table) <
      count * sizeof(iree_io_parameters_string_entry_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "key string table must have enough data to service all defined spans");
  }

  // Map string table; note that the offsets are validated during enumeration.
  iree_const_byte_span_t key_table_ptr = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(key_table, 0,
                                             iree_vm_buffer_length(key_table),
                                             sizeof(uint32_t), &key_table_ptr));
  out_args->string_table =
      (const iree_io_parameters_string_entry_t*)key_table_ptr.data;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_ro(key_data, 0, iree_vm_buffer_length(key_data),
                            sizeof(char), &out_args->string_data));

  // Map span data; the offsets/lengths are validated in the parameter provider
  // implementation.
  iree_host_size_t span_list_size =
      count * sizeof(iree_io_parameters_span_entry_t);
  iree_const_byte_span_t span_list_ptr = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(spans, 0, span_list_size,
                                             sizeof(uint64_t), &span_list_ptr));
  out_args->spans = (const iree_io_parameters_span_entry_t*)span_list_ptr.data;

  out_args->count = count;
  return iree_ok_status();
}

static iree_status_t iree_io_parameters_resolve_string(
    iree_io_parameters_string_entry_t key, iree_const_byte_span_t string_data,
    iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();

  // Check if the start of the range runs off the end of the buffer.
  if (IREE_UNLIKELY(key.offset > string_data.data_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "attempted to access an address off the end of the valid buffer range "
        "(offset=%u, length=%u, data_capacity=%" PRIhsz ")",
        key.offset, key.length, string_data.data_length);
  }

  if (key.length == 0) {
    // Fine to have a zero length.
    return iree_ok_status();
  }

  // Check if the end runs over the allocation.
  uint32_t end = key.offset + key.length;
  if (IREE_UNLIKELY(end > string_data.data_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "attempted to access an address outside of the valid buffer range "
        "(offset=%u, length=%u, end(inc)=%u, data_capacity=%" PRIhsz ")",
        key.offset, key.length, end - 1, string_data.data_length);
  }

  out_key->data = (const char*)string_data.data + key.offset;
  out_key->size = (iree_host_size_t)key.length;
  return iree_ok_status();
}

static iree_status_t iree_io_parameters_indirect_enumerator(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const iree_io_parameters_indirect_args_t* args =
      (const iree_io_parameters_indirect_args_t*)user_data;
  if (IREE_UNLIKELY(i >= args->count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter out of bounds");
  }
  IREE_RETURN_IF_ERROR(iree_io_parameters_resolve_string(
      args->string_table[i], args->string_data, out_key));
  const iree_io_parameters_span_entry_t span = args->spans[i];
  out_span->parameter_offset = span.parameter_offset;
  out_span->buffer_offset = iree_hal_cast_device_size(span.buffer_offset);
  out_span->length = iree_hal_cast_device_size(span.length);
  return iree_ok_status();
}

static iree_status_t iree_io_parameters_vm_list_emitter(
    void* user_data, iree_host_size_t i, iree_hal_buffer_t* buffer) {
  iree_vm_list_t* list = (iree_vm_list_t*)user_data;
  return iree_vm_list_set_buffer_retain(list, i, buffer);
}

//===----------------------------------------------------------------------===//
// Exported functions
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_io_parameters_module_load,     //
                   iree_io_parameters_module_state_t,  //
                   rIrrrIiirrr, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_vm_buffer_t* source_scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref_or_null(args->r4, &source_scope));
  iree_hal_queue_affinity_t target_queue_affinity =
      (iree_hal_queue_affinity_t)args->i5;
  iree_hal_memory_type_t target_memory_types = (iree_hal_memory_type_t)args->i6;
  iree_hal_buffer_usage_t target_buffer_usage =
      (iree_hal_buffer_usage_t)args->i7;
  iree_vm_buffer_t* key_table = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r8, &key_table));
  iree_vm_buffer_t* key_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r9, &key_data));
  iree_vm_buffer_t* spans = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r10, &spans));

  iree_io_parameter_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_io_parameters_module_resolve_provider(
      IREE_IO_PARAMETERS_MODULE_CAST(module),
      iree_vm_buffer_as_string(source_scope), &provider));

  iree_io_parameters_indirect_args_t enumerator_args;
  IREE_RETURN_IF_ERROR(iree_io_parameters_prepare_indirect_args(
      key_table, key_data, spans, &enumerator_args));
  iree_io_parameter_enumerator_t enumerator = {
      .fn = iree_io_parameters_indirect_enumerator,
      .user_data = &enumerator_args,
  };

  iree_vm_list_t* target_buffers = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      iree_vm_make_ref_type_def(iree_hal_buffer_type()), enumerator_args.count,
      state->host_allocator, &target_buffers));
  iree_status_t status =
      iree_vm_list_resize(target_buffers, enumerator_args.count);
  iree_io_parameter_emitter_t emitter = {
      .fn = iree_io_parameters_vm_list_emitter,
      .user_data = target_buffers,
  };

  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_params_t target_params = {
        .type = target_memory_types,
        .usage = target_buffer_usage,
        .queue_affinity = target_queue_affinity,
    };
    status = iree_io_parameter_provider_load(
        provider, device, queue_affinity,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence),
        iree_vm_buffer_as_string(source_scope), target_params,
        enumerator_args.count, enumerator, emitter);
  }

  if (iree_status_is_ok(status)) {
    rets->r0 = iree_vm_list_move_ref(target_buffers);
  } else {
    iree_vm_list_release(target_buffers);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_io_parameters_module_gather,   //
                   iree_io_parameters_module_state_t,  //
                   rIrrrrrrr, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_vm_buffer_t* source_scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref_or_null(args->r4, &source_scope));
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r5, &target_buffer));
  iree_vm_buffer_t* key_table = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r6, &key_table));
  iree_vm_buffer_t* key_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r7, &key_data));
  iree_vm_buffer_t* spans = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r8, &spans));

  iree_io_parameter_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_io_parameters_module_resolve_provider(
      IREE_IO_PARAMETERS_MODULE_CAST(module),
      iree_vm_buffer_as_string(source_scope), &provider));

  iree_io_parameters_indirect_args_t enumerator_args;
  IREE_RETURN_IF_ERROR(iree_io_parameters_prepare_indirect_args(
      key_table, key_data, spans, &enumerator_args));
  iree_io_parameter_enumerator_t enumerator = {
      .fn = iree_io_parameters_indirect_enumerator,
      .user_data = &enumerator_args,
  };
  return iree_io_parameter_provider_gather(
      provider, device, queue_affinity,
      iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence),
      iree_vm_buffer_as_string(source_scope), target_buffer,
      enumerator_args.count, enumerator);
}

IREE_VM_ABI_EXPORT(iree_io_parameters_module_scatter,  //
                   iree_io_parameters_module_state_t,  //
                   rIrrrrrrr, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r4, &source_buffer));
  iree_vm_buffer_t* target_scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref_or_null(args->r5, &target_scope));
  iree_vm_buffer_t* key_table = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r6, &key_table));
  iree_vm_buffer_t* key_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r7, &key_data));
  iree_vm_buffer_t* spans = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r8, &spans));

  iree_io_parameter_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_io_parameters_module_resolve_provider(
      IREE_IO_PARAMETERS_MODULE_CAST(module),
      iree_vm_buffer_as_string(target_scope), &provider));

  iree_io_parameters_indirect_args_t enumerator_args;
  IREE_RETURN_IF_ERROR(iree_io_parameters_prepare_indirect_args(
      key_table, key_data, spans, &enumerator_args));
  iree_io_parameter_enumerator_t enumerator = {
      .fn = iree_io_parameters_indirect_enumerator,
      .user_data = &enumerator_args,
  };
  return iree_io_parameter_provider_scatter(
      provider, device, queue_affinity,
      iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), source_buffer,
      iree_vm_buffer_as_string(target_scope), enumerator_args.count,
      enumerator);
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_io_parameters_module_exports_
// table.
static const iree_vm_native_function_ptr_t iree_io_parameters_module_funcs_[] =
    {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/io/parameters/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t
    iree_io_parameters_module_imports_[1];

static const iree_vm_native_export_descriptor_t
    iree_io_parameters_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#include "iree/modules/io/parameters/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_io_parameters_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_io_parameters_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t
    iree_io_parameters_module_descriptor_ = {
        .name = iree_string_view_literal("io_parameters"),
        .version = IREE_IO_PARAMETERS_MODULE_VERSION_LATEST,
        .attr_count = 0,
        .attrs = NULL,
        .dependency_count = 0,
        .dependencies = NULL,
        .import_count = 0,  // workaround for 0-length C struct
        .imports = iree_io_parameters_module_imports_,
        .export_count = IREE_ARRAYSIZE(iree_io_parameters_module_exports_),
        .exports = iree_io_parameters_module_exports_,
        .function_count = IREE_ARRAYSIZE(iree_io_parameters_module_funcs_),
        .functions = iree_io_parameters_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_io_parameters_module_create(
    iree_vm_instance_t* instance, iree_host_size_t provider_count,
    iree_io_parameter_provider_t* const* providers,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(!provider_count || providers);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_io_parameters_module_destroy,
      .alloc_state = iree_io_parameters_module_alloc_state,
      .free_state = iree_io_parameters_module_free_state,
      .notify = iree_io_parameters_module_notify,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_io_parameters_module_t) +
      provider_count * sizeof(iree_io_parameter_provider_t*);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_io_parameters_module_descriptor_, instance,
      host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_io_parameters_module_t* module =
      IREE_IO_PARAMETERS_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->provider_count = provider_count;
  for (iree_host_size_t i = 0; i < provider_count; ++i) {
    module->providers[i] = providers[i];
    iree_io_parameter_provider_retain(providers[i]);
  }

  *out_module = base_module;
  return iree_ok_status();
}
