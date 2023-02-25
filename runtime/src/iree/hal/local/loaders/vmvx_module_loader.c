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
#include "iree/modules/vmvx/module.h"
#include "iree/vm/bytecode/module.h"

#define IREE_VMVX_ENTRY_SIGNATURE "0rrriiiiiiiii_v"

// Index of the module in the context_modules list.
// This should always be first so that it can be overridden by user modules.
#define IREE_VMVX_MODULE_INDEX 0

//===----------------------------------------------------------------------===//
// Built-in executable helpers
//===----------------------------------------------------------------------===//

// Calls the __set_constants method in |bytecode_module| with the given
// |constants|. We wrap the data in VM buffer and require that it is not
// retained by the module; the constant values should be extracted and stored in
// globals. Fails if the constant table is not of the required size.
static iree_status_t iree_hal_vmvx_executable_set_constants(
    iree_vm_context_t* context, iree_vm_module_t* bytecode_module,
    iree_host_size_t constant_count, const uint32_t* constants,
    iree_allocator_t host_allocator) {
  // Look for the exported function. If it's not present then no constants are
  // required and if it is then we must have at least one constant.
  iree_vm_function_t set_function;
  iree_status_t status = iree_vm_module_lookup_function_by_name(
      bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("__set_constants"), &set_function);
  if (iree_status_is_not_found(status)) {
    // No constants required by the executable.
    iree_status_ignore(status);
    if (constant_count > 0) {
      // ...but we got provided some anyway.
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable has no executable-level constants "
                              "but %" PRIhsz " constants were provided",
                              constant_count);
    }
    return iree_ok_status();  // nothing to do
  } else if (!iree_status_is_ok(status)) {
    return status;
  } else if (!constant_count || !constants) {
    // Constants required but none provided.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable requires executable-level constants "
                            "but none were provided");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Wrap the constant memory in an on-stack buffer.
  iree_vm_buffer_t buffer = {{0}};
  iree_vm_buffer_initialize(
      IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      iree_make_byte_span((void*)constants,
                          constant_count * sizeof(*constants)),
      iree_allocator_null(), &buffer);

  // Setup input list.
  // TODO(benvanik): replace with direct call.
  uint8_t input_storage[64] = {0};
  iree_vm_list_t* inputs = NULL;
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_ref_type(iree_vm_buffer_type_id());
  status = iree_vm_list_initialize(
      iree_make_byte_span(input_storage, sizeof(input_storage)), &element_type,
      1, &inputs);
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(&buffer);
    status = iree_vm_list_push_ref_move(inputs, &buffer_ref);
  }

  // Copy the executable constants into the module state.
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(context, set_function,
                            IREE_VM_INVOCATION_FLAG_TRACE_INLINE,
                            /*policy=*/NULL, inputs,
                            /*outputs=*/NULL, host_allocator);
  }

  // Inputs *must* be released here as we allocated it on the stack.
  if (inputs) {
    iree_vm_list_deinitialize(inputs);
  }

  // Buffer *must* be released here since we don't control the constant
  // lifetime - this will abort if it's not.
  iree_vm_buffer_deinitialize(&buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_vmvx_worker_state_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vmvx_worker_state_t {
  // Context containing both the VMVX module and the loaded executable.
  // This context may also contain custom user modules available for the
  // generated VMVX modules to use.
  iree_vm_context_t* context;

  // Pointer into the VMVX module state for the worker context.
  // This is used to update module state directly.
  iree_vm_module_state_t* vmvx_module_state;
} iree_hal_vmvx_worker_state_t;

static iree_status_t iree_hal_vmvx_worker_state_initialize(
    iree_vm_instance_t* instance, iree_host_size_t module_count,
    iree_vm_module_t** modules, iree_vm_module_t* bytecode_module,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_vmvx_worker_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_state, 0, sizeof(*out_state));

  // Create the context unique to this worker.
  iree_vm_context_t* context = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_context_create_with_modules(
              instance, IREE_VM_CONTEXT_FLAG_NONE, module_count, modules,
              host_allocator, &context));

  // Fetch the VMVX module state so that we can quickly access it to set
  // per-call state.
  iree_vm_module_t* vmvx_module = modules[IREE_VMVX_MODULE_INDEX];
  iree_vm_module_state_t* vmvx_module_state = NULL;
  iree_status_t status = iree_vm_context_resolve_module_state(
      context, vmvx_module, &vmvx_module_state);

  // Set executable-level constants.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmvx_executable_set_constants(
        context, bytecode_module, executable_params->constant_count,
        executable_params->constants, host_allocator);
  }

  if (iree_status_is_ok(status)) {
    out_state->context = context;
    out_state->vmvx_module_state = vmvx_module_state;
  } else {
    iree_vm_context_release(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vmvx_worker_state_deinitialize(
    iree_hal_vmvx_worker_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (state->context) {
    iree_vm_context_release(state->context);
    state->context = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_vmvx_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vmvx_executable_t {
  iree_hal_local_executable_t base;

  // Loaded VMVX module shared across all workers.
  iree_vm_module_t* bytecode_module;

  // Preallocated per-worker states that are used to emulate TLS.
  iree_host_size_t worker_capacity;
  iree_hal_vmvx_worker_state_t* worker_states;

  // Resolved entry function export ordinals from the bytecode module.
  iree_host_size_t entry_fn_count;
  uint16_t entry_fn_ordinals[];
} iree_hal_vmvx_executable_t;

static const iree_hal_local_executable_vtable_t iree_hal_vmvx_executable_vtable;

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
        "convention; expected '" IREE_VMVX_ENTRY_SIGNATURE
        "' but got '%.*s', possible ABI version mismatch",
        (int)signature.calling_convention.size,
        signature.calling_convention.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vmvx_executable_create(
    iree_vm_instance_t* instance, iree_host_size_t module_count,
    iree_vm_module_t** modules, iree_vm_module_t* bytecode_module,
    iree_host_size_t worker_capacity,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(bytecode_module);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(!executable_params->pipeline_layout_count ||
                       executable_params->pipeline_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: pipeline layouts are optional but if provided must be consistent.
  iree_host_size_t entry_count =
      iree_vm_module_signature(bytecode_module).export_function_count;
  if (executable_params->pipeline_layout_count > 0 &&
      entry_count != executable_params->pipeline_layout_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_count,
                            executable_params->pipeline_layout_count);
  }

  iree_hal_vmvx_executable_t* executable = NULL;
  const iree_host_size_t entry_fn_ordinals_size =
      iree_host_align(entry_count * sizeof(*executable->entry_fn_ordinals), 8);
  const iree_host_size_t dispatch_attrs_size = iree_host_align(
      entry_count * sizeof(*executable->base.dispatch_attrs), 8);
  const iree_host_size_t pipeline_layouts_size =
      iree_host_align(executable_params->pipeline_layout_count *
                          sizeof(iree_hal_pipeline_layout_t*),
                      8);
  const iree_host_size_t worker_states_size =
      iree_host_align(worker_capacity * sizeof(*executable->worker_states), 8);
  const iree_host_size_t total_size =
      sizeof(*executable) + entry_fn_ordinals_size + dispatch_attrs_size +
      pipeline_layouts_size + worker_states_size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs = NULL;
  if (iree_status_is_ok(status)) {
    uint8_t* ptr =
        (uint8_t*)executable + sizeof(*executable) + entry_fn_ordinals_size;
    dispatch_attrs = (iree_hal_executable_dispatch_attrs_v0_t*)ptr;
    ptr += dispatch_attrs_size;
    iree_hal_pipeline_layout_t** pipeline_layouts_ptr =
        (iree_hal_pipeline_layout_t**)ptr;
    ptr += pipeline_layouts_size;
    iree_hal_local_executable_initialize(
        &iree_hal_vmvx_executable_vtable,
        executable_params->pipeline_layout_count,
        executable_params->pipeline_layouts, pipeline_layouts_ptr,
        host_allocator, &executable->base);
    executable->base.dispatch_attrs = dispatch_attrs;

    executable->worker_capacity = worker_capacity;
    executable->worker_states = (iree_hal_vmvx_worker_state_t*)ptr;
    ptr += worker_states_size;

    executable->bytecode_module = bytecode_module;
    executable->entry_fn_count = entry_count;
    for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
      iree_vm_function_t entry_fn;
      status = iree_vm_module_lookup_function_by_ordinal(
          bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &entry_fn);
      if (!iree_status_is_ok(status)) break;
      status = iree_hal_vmvx_executable_verify_entry_point(&entry_fn);
      if (!iree_status_is_ok(status)) break;
      IREE_ASSERT_EQ(entry_fn.module, bytecode_module);
      IREE_ASSERT_EQ(entry_fn.linkage, IREE_VM_FUNCTION_LINKAGE_EXPORT);
      executable->entry_fn_ordinals[i] = entry_fn.ordinal;
    }
  }

  // Query the optional local workgroup size from each entry point.
  if (iree_status_is_ok(status)) {
    // TODO(benvanik): pack this more efficiently; this requires a lot of
    // queries and instead could be a single packed table we can directly
    // reference from the module. Module-level reflection attrs would help.
    for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
      iree_vm_function_t entry_fn = {
          .module = executable->bytecode_module,
          .linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT,
          .ordinal = executable->entry_fn_ordinals[i],
      };
      iree_string_view_t local_memory_str =
          iree_vm_function_lookup_attr_by_name(
              &entry_fn, iree_make_cstring_view("local_memory"));
      uint32_t local_memory_size = 0;
      if (!iree_string_view_is_empty(local_memory_str)) {
        iree_string_view_atoi_uint32(local_memory_str, &local_memory_size);
      }
      local_memory_size /= IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE;
      dispatch_attrs[i].local_memory_pages = (uint16_t)local_memory_size;
    }
  }

  // Initialize a context per worker requested.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < worker_capacity; ++i) {
      status = iree_hal_vmvx_worker_state_initialize(
          instance, module_count, modules, bytecode_module, executable_params,
          host_allocator, &executable->worker_states[i]);
      if (!iree_status_is_ok(status)) break;
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

  for (iree_host_size_t i = 0; i < executable->worker_capacity; ++i) {
    iree_hal_vmvx_worker_state_deinitialize(&executable->worker_states[i]);
  }
  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vmvx_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  iree_hal_vmvx_executable_t* executable =
      (iree_hal_vmvx_executable_t*)base_executable;

  // Map the export ordinal to the exported function in the bytecode module.
  if (IREE_UNLIKELY(ordinal >= executable->entry_fn_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }
  iree_vm_function_t entry_fn = {
      .module = executable->bytecode_module,
      .linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT,
      .ordinal = executable->entry_fn_ordinals[ordinal],
  };

  // Fetch worker-local state. This caller is the only one able to access it so
  // no synchronization is required.
  if (IREE_UNLIKELY(worker_id >= executable->worker_capacity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "worker_id out of bounds");
  }
  iree_hal_vmvx_worker_state_t* worker_state =
      &executable->worker_states[worker_id];
  iree_vmvx_module_state_update_workgroup_state(worker_state->vmvx_module_state,
                                                workgroup_state->processor_id);

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
  IREE_RETURN_IF_ERROR(iree_vm_list_initialize(
      iree_make_byte_span(binding_list_storage, binding_list_size),
      &buffer_type, dispatch_state->binding_count, &binding_list));

  // Map bindings into on-stack VMVX buffers.
  iree_status_t status = iree_ok_status();
  iree_vm_buffer_t* binding_buffers = (iree_vm_buffer_t*)iree_alloca(
      dispatch_state->binding_count * sizeof(iree_vm_buffer_t));
  for (iree_host_size_t i = 0; i < dispatch_state->binding_count; ++i) {
    iree_vm_buffer_t* binding_buffer = &binding_buffers[i];
    // TODO(benvanik): pipeline layout contains the required access
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
    status =
        iree_vm_ref_wrap_assign(binding_buffer, iree_vm_buffer_type_id(), &ref);
    if (!iree_status_is_ok(status)) break;
    status = iree_vm_list_push_ref_retain(binding_list, &ref);
    if (!iree_status_is_ok(status)) break;
  }
  if (!iree_status_is_ok(status)) {
    iree_vm_list_deinitialize(binding_list);
    return status;
  }

  // Acquire workgroup local memory for the dispatch.
  iree_vm_buffer_t local_memory_buffer;
  iree_vm_buffer_initialize(
      IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      iree_make_byte_span(workgroup_state->local_memory,
                          workgroup_state->local_memory_size),
      iree_allocator_null(), &local_memory_buffer);

  // Map the push constant memory directly from the dispatch state.
  iree_vm_buffer_t constants_buffer;
  iree_vm_buffer_initialize(
      IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      iree_make_byte_span(
          (void*)dispatch_state->push_constants,
          sizeof(uint32_t) * dispatch_state->push_constant_count),
      iree_allocator_null(), &constants_buffer);

  // Prepare call argument buffer. We've verified the signature on creation and
  // know the exact format we can assume here.
  //
  //   func.func @entry(
  //       %local_memory: !vmvx.buffer,
  //       %constants: !vmvx.buffer,
  //       %bindings: !util.list<!vmvx.buffer>,
  //       %workgroup_id_x: i32,
  //       %workgroup_id_y: i32,
  //       %workgroup_id_z: i32,
  //       %workgroup_size_x: i32,
  //       %workgroup_size_y: i32,
  //       %workgroup_size_z: i32,
  //       %workgroup_count_x: i32,
  //       %workgroup_count_y: i32,
  //       %workgroup_count_z: i32
  //    )
  //
  // NOTE: this level of the VM ABI is supported - but may change in the future.
  // Users should prefer to use the invocation API that is more stable.
  struct {
    iree_vm_ref_t local_memory;
    iree_vm_ref_t constants;
    iree_vm_ref_t bindings;
    uint32_t workgroup_id_x;
    uint32_t workgroup_id_y;
    uint32_t workgroup_id_z;
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
      .workgroup_id_x = workgroup_state->workgroup_id_x,
      .workgroup_id_y = workgroup_state->workgroup_id_y,
      .workgroup_id_z = workgroup_state->workgroup_id_z,
      .workgroup_size_x = dispatch_state->workgroup_size_x,
      .workgroup_size_y = dispatch_state->workgroup_size_y,
      .workgroup_size_z = dispatch_state->workgroup_size_z,
      .workgroup_count_x = dispatch_state->workgroup_count_x,
      .workgroup_count_y = dispatch_state->workgroup_count_y,
      .workgroup_count_z = dispatch_state->workgroup_count_z,
  };

  // Call arguments are retained by the caller.
  iree_vm_list_retain(binding_list);            // for call
  iree_vm_buffer_retain(&local_memory_buffer);  // for call
  iree_vm_buffer_retain(&constants_buffer);     // for call

  // VM stack stored on native stack. We really do abuse the stack too much
  // here but it's 8KB and that should be reasonable given that there isn't too
  // much above us in the stack.
  // TODO(benvanik): pass in an iree_arena_t that can be used for this.
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack, IREE_VM_INVOCATION_FLAG_TRACE_INLINE,
      iree_vm_context_state_resolver(worker_state->context),
      executable->base.host_allocator);

  // Direct call interface.
  // This only works because we know the exact signature and that these will
  // never block (if they do it'll be handled as if it's an error).
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = entry_fn;
  call.arguments = iree_make_byte_span(&call_args, sizeof(call_args));
  call.results = iree_make_byte_span(NULL, 0);
  status = entry_fn.module->begin_call(entry_fn.module->self, stack, call);

  // Clean up the stack if needed, such as when the call fails.
  iree_vm_stack_deinitialize(stack);

  iree_vm_buffer_deinitialize(&local_memory_buffer);
  iree_vm_buffer_deinitialize(&constants_buffer);
  iree_vm_list_deinitialize(binding_list);
  for (iree_host_size_t i = 0; i < dispatch_state->binding_count; ++i) {
    iree_vm_buffer_deinitialize(&binding_buffers[i]);
  }

  return status;
}

static const iree_hal_local_executable_vtable_t
    iree_hal_vmvx_executable_vtable = {
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
  iree_host_size_t common_module_count;
  iree_vm_module_t* common_modules[];
} iree_hal_vmvx_module_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_vmvx_module_loader_vtable;

iree_status_t iree_hal_vmvx_module_loader_create(
    iree_vm_instance_t* instance, iree_host_size_t user_module_count,
    iree_vm_module_t** user_modules, iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(!user_module_count || user_modules);
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // A single VMVX module is shared across all loaded executables.
  iree_vm_module_t* vmvx_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vmvx_module_create(instance, host_allocator, &vmvx_module));

  iree_host_size_t common_module_count = 1 + user_module_count;
  iree_hal_vmvx_module_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(*executable_loader) +
          common_module_count * sizeof(executable_loader->common_modules[0]),
      (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_vmvx_module_loader_vtable,
        iree_hal_executable_import_provider_null(), &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    executable_loader->instance = instance;
    iree_vm_instance_retain(executable_loader->instance);

    // We prepend the vmvx_module to any user-provided modules.
    // This yields a single ordered list of modules to pass into contexts with
    // the generated module coming last so it can resolve imports from all.
    executable_loader->common_module_count = common_module_count;
    executable_loader->common_modules[IREE_VMVX_MODULE_INDEX] = vmvx_module;
    iree_vm_module_retain(vmvx_module);
    for (iree_host_size_t i = 0; i < user_module_count; ++i) {
      executable_loader->common_modules[1 + i] = user_modules[i];
      iree_vm_module_retain(user_modules[i]);
    }

    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  iree_vm_module_release(vmvx_module);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vmvx_module_loader_create_isolated(
    iree_host_size_t user_module_count, iree_vm_module_t** user_modules,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_instance_create(host_allocator, &instance));

  iree_status_t status = iree_hal_vmvx_module_loader_create(
      instance, user_module_count, user_modules, host_allocator,
      out_executable_loader);

  iree_vm_instance_release(instance);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vmvx_module_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_vmvx_module_loader_t* executable_loader =
      (iree_hal_vmvx_module_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable_loader->common_module_count;
       ++i) {
    iree_vm_module_release(executable_loader->common_modules[i]);
  }
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
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  iree_hal_vmvx_module_loader_t* executable_loader =
      (iree_hal_vmvx_module_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_const_byte_span_t bytecode_module_data =
      executable_params->executable_data;

  // If the caching mode allows for aliasing the existing FlatBuffer data then
  // we avoid allocations and just pass the pointer on through. The caller
  // ensures that the data remains valid for the duration the executable is
  // loaded. Otherwise, we clone it and let the bytecode module take ownership.
  iree_allocator_t bytecode_module_allocator;
  if (iree_all_bits_set(executable_params->caching_mode,
                        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA)) {
    // Zero-copy route.
    bytecode_module_allocator = iree_allocator_null();
  } else {
    bytecode_module_allocator = executable_loader->host_allocator;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_clone(executable_loader->host_allocator,
                                 executable_params->executable_data,
                                 (void**)&bytecode_module_data.data));
  }

  // Load the user-provided bytecode module. We pass ownership of the data (if
  // we have it) to the module to manage.
  iree_vm_module_t* bytecode_module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      executable_loader->instance, executable_params->executable_data,
      bytecode_module_allocator, executable_loader->host_allocator,
      &bytecode_module);

  // Executable takes ownership of the entire context (including the bytecode
  // module, which itself may own the underlying allocation).
  if (iree_status_is_ok(status)) {
    // Merge the context modules into a single flat list (as we have to pass
    // that down the API chain). If we had more than 2 modules this would be
    // worth fixing.
    iree_host_size_t context_module_count =
        executable_loader->common_module_count + 1;
    iree_vm_module_t** context_modules = (iree_vm_module_t**)iree_alloca(
        context_module_count * sizeof(iree_vm_module_t*));
    memcpy(context_modules, executable_loader->common_modules,
           executable_loader->common_module_count *
               sizeof(executable_loader->common_modules[0]));
    context_modules[context_module_count - 1] = bytecode_module;

    // Create the executable, including the VM contexts for each worker.
    status = iree_hal_vmvx_executable_create(
        executable_loader->instance, context_module_count, context_modules,
        bytecode_module, worker_capacity, executable_params,
        executable_loader->host_allocator, out_executable);
  }

  iree_vm_module_release(bytecode_module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_vmvx_module_loader_vtable = {
        .destroy = iree_hal_vmvx_module_loader_destroy,
        .query_support = iree_hal_vmvx_module_loader_query_support,
        .try_load = iree_hal_vmvx_module_loader_try_load,
};
