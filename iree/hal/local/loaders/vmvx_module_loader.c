// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/vmvx_module_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/local_executable_layout.h"
#include "iree/modules/vmvx/module.h"
#include "iree/vm/bytecode_module.h"

//===----------------------------------------------------------------------===//
// iree_hal_vmvx_executable_t
//===----------------------------------------------------------------------===//

#define IREE_VMVX_ENTRY_SIGNATURE "0rrriiiiiiiii_v"

typedef struct iree_hal_vmvx_executable_t {
  iree_hal_local_executable_t base;

  // Context containing both the VMVX module and the loaded executable.
  iree_vm_context_t* context;

  // Resolved entry functions from the module.
  iree_host_size_t entry_fn_count;
  iree_vm_function_t entry_fns[];
} iree_hal_vmvx_executable_t;

extern const iree_hal_local_executable_vtable_t iree_hal_vmvx_executable_vtable;

// Verifies that an entry point function exported by the bytecode module matches
// the calling convention we expect. This avoids the need to check it during
// dispatch (where returning errors is hard and it'd be expensive).
static iree_status_t iree_hal_vmvx_executable_verify_entry_point(
    iree_vm_function_t* entry_fn) {
  iree_vm_function_signature_t signature = iree_vm_function_signature(entry_fn);
  if (!iree_string_view_equal(
          signature.calling_convention,
          iree_make_cstring_view(IREE_VMVX_ENTRY_SIGNATURE))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable entry point does not match the expected calling "
        "convention; expected '" IREE_VMVX_ENTRY_SIGNATURE "' but got '%.*s'",
        (int)signature.calling_convention.size,
        signature.calling_convention.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vmvx_executable_create(
    iree_vm_context_t* context, iree_vm_module_t* bytecode_module,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bytecode_module);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t entry_count =
      iree_vm_module_signature(bytecode_module).export_function_count;
  if (entry_count != executable_layout_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_count, executable_layout_count);
  }

  iree_hal_vmvx_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) + entry_count * sizeof(*executable->entry_fns) +
      entry_count * sizeof(*executable->base.dispatch_attrs) +
      executable_layout_count * sizeof(iree_hal_local_executable_layout_t);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs = NULL;
  if (iree_status_is_ok(status)) {
    uint8_t* ptr = (uint8_t*)executable + sizeof(*executable) +
                   entry_count * sizeof(*executable->entry_fns);
    dispatch_attrs = (iree_hal_executable_dispatch_attrs_v0_t*)ptr;
    ptr += entry_count * sizeof(*executable->base.dispatch_attrs);
    iree_hal_local_executable_layout_t** executable_layouts_ptr =
        (iree_hal_local_executable_layout_t**)ptr;
    iree_hal_local_executable_initialize(
        &iree_hal_vmvx_executable_vtable, executable_layout_count,
        executable_layouts, executable_layouts_ptr, host_allocator,
        &executable->base);
    executable->context = context;
    executable->base.dispatch_attrs = dispatch_attrs;
    iree_vm_context_retain(executable->context);

    executable->entry_fn_count = entry_count;
    for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
      status = iree_vm_module_lookup_function_by_ordinal(
          bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i,
          &executable->entry_fns[i], NULL);
      if (!iree_status_is_ok(status)) break;
      status = iree_hal_vmvx_executable_verify_entry_point(
          &executable->entry_fns[i]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Query the optional local workgroup size from each entry point.
  if (iree_status_is_ok(status)) {
    // TODO(benvanik): pack this more efficiently; this requires a lot of
    // queries and instead could be a single packed table we can directly
    // reference from the module. Module-level reflection attrs would help.
    for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
      iree_string_view_t local_memory_str = iree_vm_function_reflection_attr(
          &executable->entry_fns[i], iree_make_cstring_view("local_memory"));
      uint32_t local_memory_size = 0;
      if (!iree_string_view_is_empty(local_memory_str)) {
        iree_string_view_atoi_uint32(local_memory_str, &local_memory_size);
      }
      local_memory_size /= IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE;
      dispatch_attrs[i].local_memory_pages = (uint16_t)local_memory_size;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vmvx_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_vmvx_executable_t* executable =
      (iree_hal_vmvx_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_context_release(executable->context);
  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vmvx_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id, iree_byte_span_t local_memory) {
  iree_hal_vmvx_executable_t* executable =
      (iree_hal_vmvx_executable_t*)base_executable;

  if (IREE_UNLIKELY(ordinal >= executable->entry_fn_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }
  iree_vm_function_t entry_fn = executable->entry_fns[ordinal];

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_string_view_t entry_point_name = iree_vm_function_name(&entry_fn);
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_vmvx_call");
  }
  IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, entry_point_name.data,
                                      entry_point_name.size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  // On-stack interface local to this invocation.
  // Note that we _could_ share this across all invocations in a dispatch, but
  // it's tricky to find a good place when threading is happening and it's
  // intentionally fairly cheap to construct by matching the dispatch_state.
  // The list would only need to be constructed once and we could avoid the
  // extraneous retain/releases and mappings.
  iree_vm_type_def_t buffer_type =
      iree_vm_type_def_make_ref_type(iree_vm_buffer_type_id());
  iree_host_size_t binding_list_size =
      iree_vm_list_storage_size(&buffer_type, dispatch_state->binding_count);
  void* binding_list_storage = iree_alloca(binding_list_size);
  iree_vm_list_t* binding_list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_initialize(
              iree_make_byte_span(binding_list_storage, binding_list_size),
              &buffer_type, dispatch_state->binding_count, &binding_list));
  iree_vm_list_retain(binding_list);  // for call

  // Map bindings into on-stack VMVX buffers.
  iree_vm_buffer_t* binding_buffers = (iree_vm_buffer_t*)iree_alloca(
      dispatch_state->binding_count * sizeof(iree_vm_buffer_t));
  for (iree_host_size_t i = 0; i < dispatch_state->binding_count; ++i) {
    iree_vm_buffer_t* binding_buffer = &binding_buffers[i];
    // TODO(benvanik): executable layout contains the required access
    // information. We will likely want to encode a bitmap of mutable bindings
    // such that we can quickly set the access bit, though.
    iree_vm_buffer_access_t access =
        IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST;
    iree_vm_buffer_initialize(
        access,
        iree_make_byte_span(dispatch_state->binding_ptrs[i],
                            dispatch_state->binding_lengths[i]),
        iree_allocator_null(), binding_buffer);
    iree_vm_ref_t ref = {0};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_ref_wrap_assign(binding_buffer, iree_vm_buffer_type_id(),
                                    &ref));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_push_ref_retain(binding_list, &ref));
  }

  // Acquire workgroup local memory for the dispatch.
  iree_vm_buffer_t local_memory_buffer;
  iree_vm_buffer_initialize(
      IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      local_memory, iree_allocator_null(), &local_memory_buffer);
  iree_vm_buffer_retain(&local_memory_buffer);  // for call

  // Map the push constant memory directly from the dispatch state.
  iree_vm_buffer_t constants_buffer;
  iree_vm_buffer_initialize(
      IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      iree_make_byte_span(
          (void*)dispatch_state->push_constants,
          sizeof(uint32_t) * dispatch_state->push_constant_count),
      iree_allocator_null(), &constants_buffer);
  iree_vm_buffer_retain(&constants_buffer);  // for call

  // Prepare call argument buffer. We've verified the signature on creation and
  // know the exact format we can assume here.
  //
  //   func @entry(
  //       %local_memory: !vmvx.buffer,
  //       %constants: !vmvx.buffer,
  //       %bindings: !iree.list<!vmvx.buffer>,
  //       %workgroup_x: index,
  //       %workgroup_y: index,
  //       %workgroup_z: index,
  //       %workgroup_size_x: index,
  //       %workgroup_size_y: index,
  //       %workgroup_size_z: index,
  //       %workgroup_count_x: index,
  //       %workgroup_count_y: index,
  //       %workgroup_count_z: index
  //    )
  //
  // NOTE: this level of the VM ABI is supported - but may change in the future.
  // Users should prefer to use the invocation API that is more stable.
  struct {
    iree_vm_ref_t local_memory;
    iree_vm_ref_t constants;
    iree_vm_ref_t bindings;
    uint32_t workgroup_x;
    uint32_t workgroup_y;
    uint32_t workgroup_z;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    uint32_t workgroup_count_x;
    uint32_t workgroup_count_y;
    uint32_t workgroup_count_z;
  } call_args = {
      .local_memory =
          {
              .type = iree_vm_buffer_type_id(),
              .ptr = &local_memory_buffer,
              .offsetof_counter = 0,
          },
      .constants =
          {
              .type = iree_vm_buffer_type_id(),
              .ptr = &constants_buffer,
              .offsetof_counter = 0,
          },
      .bindings =
          {
              .type = iree_vm_list_type_id(),
              .ptr = binding_list,
              .offsetof_counter = 0,
          },
      .workgroup_x = workgroup_id->x,
      .workgroup_y = workgroup_id->y,
      .workgroup_z = workgroup_id->z,
      .workgroup_size_x = dispatch_state->workgroup_size.x,
      .workgroup_size_y = dispatch_state->workgroup_size.y,
      .workgroup_size_z = dispatch_state->workgroup_size.z,
      .workgroup_count_x = dispatch_state->workgroup_count.x,
      .workgroup_count_y = dispatch_state->workgroup_count.y,
      .workgroup_count_z = dispatch_state->workgroup_count.z,
  };

  // On-stack stack. We really do abuse the stack too much here.
  // TODO(benvanik): pass in an iree_arena_t that can be used for this.
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack, iree_vm_context_state_resolver(executable->context),
      executable->base.host_allocator);

  // Direct call interface.
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = entry_fn;
  call.arguments = iree_make_byte_span(&call_args, sizeof(call_args));
  call.results = iree_make_byte_span(NULL, 0);
  iree_vm_execution_result_t result;
  iree_status_t status =
      entry_fn.module->begin_call(entry_fn.module->self, stack, &call, &result);

  iree_vm_stack_deinitialize(stack);

  iree_vm_buffer_deinitialize(&local_memory_buffer);
  iree_vm_buffer_deinitialize(&constants_buffer);
  iree_vm_list_deinitialize(binding_list);
  for (iree_host_size_t i = 0; i < dispatch_state->binding_count; ++i) {
    iree_vm_buffer_deinitialize(&binding_buffers[i]);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_local_executable_vtable_t iree_hal_vmvx_executable_vtable = {
    .base =
        {
            .destroy = iree_hal_vmvx_executable_destroy,
        },
    .issue_call = iree_hal_vmvx_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_vmvx_module_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vmvx_module_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  iree_vm_instance_t* instance;
  iree_vm_module_t* vmvx_module;
} iree_hal_vmvx_module_loader_t;

extern const iree_hal_executable_loader_vtable_t
    iree_hal_vmvx_module_loader_vtable;

iree_status_t iree_hal_vmvx_module_loader_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // A single VMVX module is shared across all loaded executables.
  IREE_RETURN_IF_ERROR(iree_vmvx_module_register_types());
  iree_vm_module_t* vmvx_module = NULL;
  IREE_RETURN_IF_ERROR(iree_vmvx_module_create(host_allocator, &vmvx_module));

  iree_hal_vmvx_module_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_vmvx_module_loader_vtable,
        iree_hal_executable_import_provider_null(), &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    executable_loader->instance = instance;
    iree_vm_instance_retain(executable_loader->instance);
    executable_loader->vmvx_module = vmvx_module;
    iree_vm_module_retain(executable_loader->vmvx_module);
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  iree_vm_module_release(vmvx_module);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vmvx_module_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_vmvx_module_loader_t* executable_loader =
      (iree_hal_vmvx_module_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_module_release(executable_loader->vmvx_module);
  iree_vm_instance_release(executable_loader->instance);
  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_vmvx_module_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("vmvx-bytecode-fb"));
}

static iree_status_t iree_hal_vmvx_module_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_vmvx_module_loader_t* executable_loader =
      (iree_hal_vmvx_module_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_const_byte_span_t bytecode_module_data =
      executable_spec->executable_data;

  // If the caching mode allows for aliasing the existing flatbuffer data then
  // we avoid allocations and just pass the pointer on through. The caller
  // ensures that the data remains valid for the duration the executable is
  // loaded. Otherwise, we clone it and let the bytecode module take ownership.
  iree_allocator_t bytecode_module_allocator;
  if (iree_all_bits_set(executable_spec->caching_mode,
                        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA)) {
    // Zero-copy route.
    bytecode_module_allocator = iree_allocator_null();
  } else {
    bytecode_module_allocator = executable_loader->host_allocator;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_clone(executable_loader->host_allocator,
                                 executable_spec->executable_data,
                                 (void**)&bytecode_module_data.data));
  }

  // Load the user-provided bytecode module. We pass ownership of the data (if
  // we have it) to the module to manage.
  iree_vm_module_t* bytecode_module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      executable_spec->executable_data, bytecode_module_allocator,
      executable_loader->host_allocator, &bytecode_module);

  // Create the context tying together the shared VMVX module and the
  // user-provided module that references it. If we wanted to allow custom
  // modules here for user-provided functions we'd mix them in here.
  iree_vm_context_t* context = NULL;
  if (iree_status_is_ok(status)) {
    iree_vm_module_t* modules[2] = {
        executable_loader->vmvx_module,
        bytecode_module,
    };
    status = iree_vm_context_create_with_modules(
        executable_loader->instance, modules, IREE_ARRAYSIZE(modules),
        executable_loader->host_allocator, &context);
  }

  // Executable takes ownership of the entire context (including the bytecode
  // module, which itself may own the underlying allocation).
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmvx_executable_create(
        context, bytecode_module, executable_spec->executable_layout_count,
        executable_spec->executable_layouts, executable_loader->host_allocator,
        out_executable);
  }

  iree_vm_context_release(context);
  iree_vm_module_release(bytecode_module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_executable_loader_vtable_t iree_hal_vmvx_module_loader_vtable = {
    .destroy = iree_hal_vmvx_module_loader_destroy,
    .query_support = iree_hal_vmvx_module_loader_query_support,
    .try_load = iree_hal_vmvx_module_loader_try_load,
};
