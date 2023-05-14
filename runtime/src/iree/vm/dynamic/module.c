// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/dynamic/module.h"

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/tracing.h"

typedef struct {
  // Interface containing local function pointers.
  // interface.self will be the self pointer to iree_vm_dynamic_module_t.
  //
  // Must be first in the struct as we dereference the interface to find our
  // members below.
  iree_vm_module_t interface;

  // The user module loaded from the dynamic library.
  // We hide this so that the dynamic module is exposed to the rest of the VM
  // and can intercept all calls for tracing/diagnostics/debugging. There should
  // be no other retainers of the user module and when we drop our reference
  // we can unload the library containing it.
  iree_vm_module_t* user_module;

  // Allocator this module was allocated with and must be freed with.
  iree_allocator_t allocator;

  // Loaded system library containing the module.
  iree_dynamic_library_t* handle;
} iree_vm_dynamic_module_t;

static iree_status_t iree_vm_dynamic_module_instantiate(
    iree_vm_dynamic_module_t* module, iree_vm_instance_t* instance,
    iree_string_view_t export_name, iree_host_size_t param_count,
    const iree_string_pair_t* params) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, export_name.data, export_name.size);

  // Lookup the exported function used to create the module.
  char export_name_str[256] = {0};
  iree_string_view_to_cstring(export_name, export_name_str,
                              sizeof(export_name_str));
  iree_vm_dynamic_module_create_fn_t create_fn = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_lookup_symbol(module->handle, export_name_str,
                                             (void**)&create_fn));

  // Try to create the module, which may fail if the version is incompatible or
  // the parameters are invalid.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      create_fn(IREE_VM_DYNAMIC_MODULE_VERSION_LATEST, instance, param_count,
                params, module->allocator, &module->user_module));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR iree_vm_dynamic_module_destroy(void* self) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  iree_allocator_t allocator = module->allocator;

  // Destroy the user module - we should be the last retainer of it since we've
  // never exposed its pointer out to the rest of the system.
  iree_vm_module_release(module->user_module);
  module->user_module = NULL;

  // Unload the dynamic library now that we know nothing should be using code
  // from it.
  iree_dynamic_library_release(module->handle);

  iree_allocator_free(allocator, module);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t IREE_API_PTR iree_vm_dynamic_module_name(void* self) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->name(module->user_module->self);
}

static iree_vm_module_signature_t IREE_API_PTR
iree_vm_dynamic_module_signature(void* self) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->signature(module->user_module->self);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_get_module_attr(
    void* self, iree_host_size_t index, iree_string_pair_t* out_attr) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->get_module_attr(module->user_module->self, index,
                                              out_attr);
}

static iree_status_t iree_vm_dynamic_module_enumerate_dependencies(
    void* self, iree_vm_module_dependency_callback_t callback,
    void* user_data) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->enumerate_dependencies(module->user_module->self,
                                                     callback, user_data);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_get_function(
    void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  IREE_RETURN_IF_ERROR(module->user_module->get_function(
      module->user_module->self, linkage, ordinal, out_function, out_name,
      out_signature));
  if (out_function) out_function->module = (iree_vm_module_t*)self;
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_get_function_attr(
    void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
    iree_host_size_t index, iree_string_pair_t* out_attr) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->get_function_attr(
      module->user_module->self, linkage, ordinal, index, out_attr);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_lookup_function(
    void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
    const iree_vm_function_signature_t* expected_signature,
    iree_vm_function_t* out_function) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  IREE_RETURN_IF_ERROR(module->user_module->lookup_function(
      module->user_module->self, linkage, name, expected_signature,
      out_function));
  out_function->module = (iree_vm_module_t*)self;
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR
iree_vm_dynamic_module_resolve_source_location(
    void* self, iree_vm_function_t function, iree_vm_source_offset_t pc,
    iree_vm_source_location_t* out_source_location) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->resolve_source_location(
      module->user_module->self, function, pc, out_source_location);
}

static iree_status_t IREE_API_PTR
iree_vm_dynamic_module_alloc_state(void* self, iree_allocator_t allocator,
                                   iree_vm_module_state_t** out_module_state) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->alloc_state(module->user_module->self, allocator,
                                          out_module_state);
}

static void IREE_API_PTR iree_vm_dynamic_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  module->user_module->free_state(module->user_module->self, module_state);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_resolve_import(
    void* self, iree_vm_module_state_t* module_state, iree_host_size_t ordinal,
    const iree_vm_function_t* function,
    const iree_vm_function_signature_t* signature) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->resolve_import(
      module->user_module->self, module_state, ordinal, function, signature);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->notify(module->user_module->self, module_state,
                                     signal);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_begin_call(
    void* self, iree_vm_stack_t* stack, iree_vm_function_call_t call) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->begin_call(module->user_module->self, stack,
                                         call);
}

static iree_status_t IREE_API_PTR iree_vm_dynamic_module_resume_call(
    void* self, iree_vm_stack_t* stack, iree_byte_span_t call_results) {
  iree_vm_dynamic_module_t* module = (iree_vm_dynamic_module_t*)self;
  return module->user_module->resume_call(module->user_module->self, stack,
                                          call_results);
}

static iree_status_t iree_vm_dynamic_module_create(
    iree_vm_instance_t* instance, iree_dynamic_library_t* handle,
    iree_string_view_t export_name, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create the wrapper and stash the library handle.
  // This makes cleanup easier if we fail below.
  iree_vm_dynamic_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  module->allocator = allocator;
  module->handle = handle;
  iree_dynamic_library_retain(module->handle);

  // Instantiate the loaded module implementation.
  iree_status_t status = iree_vm_dynamic_module_instantiate(
      module, instance, export_name, param_count, params);

  // Populate base interface to route to our thunks.
  iree_vm_module_initialize(&module->interface, module);
  module->interface.destroy = iree_vm_dynamic_module_destroy;
  module->interface.name = iree_vm_dynamic_module_name;
  module->interface.signature = iree_vm_dynamic_module_signature;
  module->interface.enumerate_dependencies =
      iree_vm_dynamic_module_enumerate_dependencies;
  module->interface.get_module_attr = iree_vm_dynamic_module_get_module_attr;
  module->interface.lookup_function = iree_vm_dynamic_module_lookup_function;
  module->interface.get_function = iree_vm_dynamic_module_get_function;
  module->interface.get_function_attr =
      iree_vm_dynamic_module_get_function_attr;
  module->interface.resolve_source_location =
      iree_vm_dynamic_module_resolve_source_location;
  module->interface.alloc_state = iree_vm_dynamic_module_alloc_state;
  module->interface.free_state = iree_vm_dynamic_module_free_state;
  module->interface.resolve_import = iree_vm_dynamic_module_resolve_import;
  module->interface.notify = iree_vm_dynamic_module_notify;
  module->interface.begin_call = iree_vm_dynamic_module_begin_call;
  module->interface.resume_call = iree_vm_dynamic_module_resume_call;

  if (iree_status_is_ok(status)) {
    *out_module = (iree_vm_module_t*)module;
  } else {
    iree_vm_module_release((iree_vm_module_t*)module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_dynamic_module_load_from_file(
    iree_vm_instance_t* instance, iree_string_view_t path,
    iree_string_view_t export_name, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(!param_count || params);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  // Default name for the export.
  if (iree_string_view_is_empty(export_name)) {
    export_name = iree_make_cstring_view(IREE_VM_DYNAMIC_MODULE_EXPORT_NAME);
  }
  IREE_TRACE_ZONE_APPEND_TEXT(z0, export_name.data, export_name.size);

  // Try to load the library from file using the system loader.
  // This can fail for many reasons (file not found, not accessible,
  // incompatible, unsigned code, missing required imports, etc).
  char path_str[2048] = {0};
  iree_string_view_to_cstring(path, path_str, sizeof(path_str));
  iree_dynamic_library_t* handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_load_from_file(
              path_str, IREE_DYNAMIC_LIBRARY_FLAG_NONE, allocator, &handle));

  // Create the module wrapper and then ask the library for its
  // implementation.
  iree_status_t status =
      iree_vm_dynamic_module_create(instance, handle, export_name, param_count,
                                    params, allocator, out_module);

  iree_dynamic_library_release(handle);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
