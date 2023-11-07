// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/loader/module.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/local_executable.h"
#include "iree/vm/api.h"

#define IREE_HAL_LOADER_MODULE_VERSION_0_0 0x00000000u
#define IREE_HAL_LOADER_MODULE_VERSION_LATEST IREE_HAL_LOADER_MODULE_VERSION_0_0

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_hal_loader_module_t {
  iree_allocator_t host_allocator;
  iree_hal_loader_module_flags_t flags;
  // TODO(benvanik): types.
  iree_host_size_t loader_count;
  iree_hal_executable_loader_t* loaders[];
} iree_hal_loader_module_t;

#define IREE_HAL_LOADER_MODULE_CAST(module)        \
  (iree_hal_loader_module_t*)((uint8_t*)(module) + \
                              iree_vm_native_module_size());

typedef struct iree_hal_loader_module_state_t {
  iree_allocator_t host_allocator;
  iree_hal_loader_module_flags_t flags;
} iree_hal_loader_module_state_t;

static void IREE_API_PTR iree_hal_loader_module_destroy(void* base_module) {
  iree_hal_loader_module_t* module = IREE_HAL_LOADER_MODULE_CAST(base_module);
  for (iree_host_size_t i = 0; i < module->loader_count; ++i) {
    iree_hal_executable_loader_release(module->loaders[i]);
  }
}

static iree_status_t IREE_API_PTR
iree_hal_loader_module_alloc_state(void* self, iree_allocator_t host_allocator,
                                   iree_vm_module_state_t** out_module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_loader_module_t* module = IREE_HAL_LOADER_MODULE_CAST(self);
  iree_hal_loader_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->flags = module->flags;

  *out_module_state = (iree_vm_module_state_t*)state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR iree_hal_loader_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_loader_module_state_t* state =
      (iree_hal_loader_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_hal_loader_module_notify(
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

// Casts a VM value to a C host size.
static iree_host_size_t iree_hal_cast_host_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if host
  // size is 32-bits.
  return (iree_host_size_t)value;
}

//===----------------------------------------------------------------------===//
// Shared argument shims
//===----------------------------------------------------------------------===//

#define IREE_HAL_ABI_EXPORT(function_name, arg_types, ret_types)               \
  IREE_VM_ABI_EXPORT(function_name, iree_hal_loader_module_state_t, arg_types, \
                     ret_types)
#define IREE_HAL_ABI_FIXED_STRUCT(name, types, body) \
  IREE_VM_ABI_FIXED_STRUCT(name, body)
#define IREE_HAL_ABI_DEFINE_SHIM(arg_types, ret_types) \
  static IREE_VM_ABI_DEFINE_SHIM(arg_types, ret_types)

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

IREE_HAL_ABI_EXPORT(iree_hal_loader_module_executable_query_support,  //
                    r, i) {
  iree_vm_buffer_t* executable_format = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref(args->r0, &executable_format));
  iree_string_view_t executable_format_str =
      iree_vm_buffer_as_string(executable_format);

  bool has_support = false;
  iree_hal_loader_module_t* loader_module = IREE_HAL_LOADER_MODULE_CAST(module);
  for (iree_host_size_t i = 0; i < loader_module->loader_count; ++i) {
    iree_hal_executable_loader_t* loader = loader_module->loaders[i];
    if (iree_hal_executable_loader_query_support(loader, 0,
                                                 executable_format_str)) {
      has_support = true;
      break;
    }
  }

  rets->i0 = has_support ? 1 : 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_loader_module_try_load(
    iree_hal_loader_module_t* loader_module,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  for (iree_host_size_t i = 0; i < loader_module->loader_count; ++i) {
    iree_hal_executable_loader_t* loader = loader_module->loaders[i];
    if (!iree_hal_executable_loader_query_support(
            loader, executable_params->caching_mode,
            executable_params->executable_format)) {
      // Loader definitely can't handle the executable; no use trying so skip.
      continue;
    }
    // The loader _may_ handle the executable; if the specific executable is not
    // supported then the try will fail with IREE_STATUS_CANCELLED and we should
    // continue trying other loaders.
    iree_status_t status = iree_hal_executable_loader_try_load(
        loader, executable_params, /*worker_capacity=*/1, out_executable);
    if (iree_status_is_ok(status)) {
      // Executable was successfully loaded.
      return status;
    } else if (!iree_status_is_cancelled(status)) {
      // Error beyond just the try failing due to unsupported formats.
      return status;
    }
    iree_status_ignore(status);
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "no executable loader registered for the given executable format '%.*s'",
      (int)executable_params->executable_format.size,
      executable_params->executable_format.data);
}

IREE_HAL_ABI_EXPORT(iree_hal_loader_module_executable_load,  //
                    rrr, r) {
  iree_vm_buffer_t* executable_format = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref(args->r0, &executable_format));
  iree_string_view_t executable_format_str =
      iree_vm_buffer_as_string(executable_format);
  iree_vm_buffer_t* executable_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &executable_data));
  iree_host_size_t constant_count = 0;
  const uint32_t* constants = NULL;
  if (iree_vm_buffer_isa(args->r2)) {
    iree_vm_buffer_t* constant_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_check_deref(args->r2, &constant_buffer));
    if (constant_buffer->data.data_length % 4 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant buffer data must contain 4-byte "
                              "elements but data length is %" PRIhsz,
                              constant_buffer->data.data_length);
    }
    constant_count = constant_buffer->data.data_length / sizeof(uint32_t);
    constants = (const uint32_t*)constant_buffer->data.data;
  }

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode |=
      executable_data->access == IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE
          ? IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA
          : 0;
  executable_params.executable_format = executable_format_str;
  executable_params.executable_data = iree_make_const_byte_span(
      executable_data->data.data, executable_data->data.data_length);
  executable_params.pipeline_layout_count = 0;
  executable_params.pipeline_layouts = NULL;
  executable_params.constant_count = constant_count;
  executable_params.constants = constants;

  iree_hal_executable_t* executable = NULL;
  iree_hal_loader_module_t* loader_module = IREE_HAL_LOADER_MODULE_CAST(module);
  iree_status_t status = iree_hal_loader_module_try_load(
      loader_module, &executable_params, &executable);

  rets->r0 = iree_hal_executable_move_ref(executable);
  return status;
}

typedef struct {
  union {
    struct {
      iree_vm_ref_t executable;
      int32_t entry_point;
      int32_t workgroup_x;
      int32_t workgroup_y;
      int32_t workgroup_z;
    };
    iree_vm_abi_riiii_t params;
  };
  iree_vm_size_t push_constant_count;
  const uint32_t* push_constants;
  iree_vm_size_t binding_count;
  const iree_vm_abi_rII_t* bindings;
} iree_hal_loader_dispatch_args_t;

static iree_status_t iree_hal_loader_module_executable_dispatch(
    iree_vm_stack_t* IREE_RESTRICT stack, void* IREE_RESTRICT module,
    iree_hal_loader_module_state_t* IREE_RESTRICT state,
    const iree_hal_loader_dispatch_args_t* IREE_RESTRICT args) {
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_check_deref(args->executable, &executable));

  if (args->binding_count > 32) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many bindings");
  }
  void** binding_ptrs =
      (void**)iree_alloca(args->binding_count * sizeof(void*));
  size_t* binding_lengths =
      (size_t*)iree_alloca(args->binding_count * sizeof(size_t));
  for (iree_host_size_t i = 0; i < args->binding_count; ++i) {
    iree_vm_buffer_t* buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_check_deref(args->bindings[i].r0, &buffer));
    // TODO(benvanik): this is a hack around not having the access permissions
    // currently modeled. This is only used for verification and early errors
    // and not intended to be a last-line defense against writes (you need an
    // MMU for that) so it's just subpar reporting.
    iree_const_byte_span_t span;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(
        buffer, iree_hal_cast_host_size(args->bindings[i].i1),
        iree_hal_cast_host_size(args->bindings[i].i2), /*alignment=*/1, &span));
    binding_ptrs[i] = (void*)span.data;
    binding_lengths[i] = span.data_length;
  }

  const iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_size_x = 1,
      .workgroup_size_y = 1,
      .workgroup_size_z = 1,
      .push_constant_count = args->push_constant_count,
      .workgroup_count_x = args->workgroup_x,
      .workgroup_count_y = args->workgroup_y,
      .workgroup_count_z = args->workgroup_z,
      .max_concurrency = 1,
      .binding_count = args->binding_count,
      .push_constants = args->push_constants,
      .binding_ptrs = binding_ptrs,
      .binding_lengths = binding_lengths,
  };

  // TODO(benvanik): environmental information.
  uint32_t processor_id = 0;
  iree_byte_span_t local_memory = iree_byte_span_empty();

  return iree_hal_local_executable_issue_dispatch_inline(
      (iree_hal_local_executable_t*)executable, args->entry_point,
      &dispatch_state, processor_id, local_memory);
}

static iree_status_t iree_vm_shim_dispatch_v(
    iree_vm_stack_t* IREE_RESTRICT stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state) {
  // TODO(benvanik): support multiple variadic segments in one call.
  // For now we inline what it would do in a very painful way.
  bool args_ok = true;
  if (args_storage.data_length <
      (sizeof(iree_vm_abi_riiii_t) + sizeof(iree_vm_size_t) +
       sizeof(iree_vm_size_t))) {
    // Can't fit even with zero lengths.
    args_ok = false;
  }
  iree_hal_loader_dispatch_args_t args = {
      .params = *(const iree_vm_abi_riiii_t*)args_storage.data,
  };
  if (args_ok) {
    const uint8_t* push_constants_ptr = args_storage.data + sizeof(args.params);
    args.push_constant_count = *(const iree_vm_size_t*)push_constants_ptr;
    args.push_constants =
        (const uint32_t*)(push_constants_ptr + sizeof(iree_vm_size_t));
    const uint8_t* bindings_ptr =
        push_constants_ptr + sizeof(iree_vm_size_t) +
        args.push_constant_count * sizeof(args.push_constants[0]);
    args.binding_count = *(const iree_vm_size_t*)bindings_ptr;
    args.bindings =
        (const iree_vm_abi_rII_t*)(bindings_ptr + sizeof(iree_vm_size_t));
    const uint8_t* max_ptr = (const uint8_t*)args.bindings +
                             args.binding_count * sizeof(args.bindings[0]);
    const uint8_t* end_ptr = args_storage.data + args_storage.data_length;
    if (max_ptr > end_ptr) args_ok = false;
  }
  if (IREE_UNLIKELY(!args_ok || rets_storage.data_length > 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }
  return iree_hal_loader_module_executable_dispatch(stack, module, module_state,
                                                    &args);
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_hal_loader_module_exports_
// table.
static const iree_vm_native_function_ptr_t iree_hal_loader_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, shim_arg_type, arg_types, ret_types) \
  {                                                                     \
      .shim = (iree_vm_native_function_shim_t)                          \
          iree_vm_shim_##shim_arg_type##_##ret_types,                   \
      .target = (iree_vm_native_function_target_t)(target_fn),          \
  },
#include "iree/modules/hal/loader/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t
    iree_hal_loader_module_imports_[1];

static const iree_vm_native_export_descriptor_t
    iree_hal_loader_module_exports_[] = {
#define EXPORT_FN(name, target_fn, shim_arg_type, arg_types, ret_types) \
  {                                                                     \
      .local_name = iree_string_view_literal(name),                     \
      .calling_convention =                                             \
          iree_string_view_literal("0" #arg_types "_" #ret_types),      \
      .attr_count = 0,                                                  \
      .attrs = NULL,                                                    \
  },
#include "iree/modules/hal/loader/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_hal_loader_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_hal_loader_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t
    iree_hal_loader_module_descriptor_ = {
        .name = iree_string_view_literal("hal_loader"),
        .version = IREE_HAL_LOADER_MODULE_VERSION_LATEST,
        .attr_count = 0,
        .attrs = NULL,
        .dependency_count = 0,
        .dependencies = NULL,
        .import_count = 0,  // workaround for 0-length C struct
        .imports = iree_hal_loader_module_imports_,
        .export_count = IREE_ARRAYSIZE(iree_hal_loader_module_exports_),
        .exports = iree_hal_loader_module_exports_,
        .function_count = IREE_ARRAYSIZE(iree_hal_loader_module_funcs_),
        .functions = iree_hal_loader_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_hal_loader_module_create(
    iree_vm_instance_t* instance, iree_hal_loader_module_flags_t flags,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_hal_loader_module_destroy,
      .alloc_state = iree_hal_loader_module_alloc_state,
      .free_state = iree_hal_loader_module_free_state,
      .notify = iree_hal_loader_module_notify,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_hal_loader_module_t) +
      loader_count * sizeof(iree_hal_executable_loader_t*);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_hal_loader_module_descriptor_, instance, host_allocator,
      base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_hal_loader_module_t* module = IREE_HAL_LOADER_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->flags = flags;
  module->loader_count = loader_count;
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    module->loaders[i] = loaders[i];
    iree_hal_executable_loader_retain(loaders[i]);
  }

  *out_module = base_module;
  return iree_ok_status();
}
