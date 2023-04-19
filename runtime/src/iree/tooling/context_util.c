// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/context_util.h"

#include <memory.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/path.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "iree/modules/hal/inline/module.h"
#include "iree/modules/hal/loader/module.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/dynamic/module.h"

#if defined(IREE_HAVE_VMVX_MODULE)
#include "iree/modules/vmvx/module.h"
#endif  // IREE_HAVE_VMVX_MODULE

//===----------------------------------------------------------------------===//
// Module loading
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, module,
    "A VM module to load; either a vmfb containing a compiled bytecode module\n"
    "or a native system library containing a dynamic native module. Modules\n"
    "are registered in the order defined by the flags with all dependencies\n"
    "for a module needing to have been registered prior to the dependent\n"
    "module. HAL modules are added automatically when required.");

static iree_status_t iree_tooling_load_bytecode_module(
    iree_vm_instance_t* instance, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  // Fetch the file contents into memory.
  // We could map the memory here if we wanted to and were coming from a file
  // on disk.
  iree_file_contents_t* file_contents = NULL;
  if (iree_string_view_equal(path, IREE_SV("-"))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_stdin_read_contents(host_allocator, &file_contents));
  } else {
    char path_str[2048] = {0};
    iree_string_view_to_cstring(path, path_str, sizeof(path_str));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_file_read_contents(path_str, host_allocator, &file_contents));
  }

  // Try to load the module as bytecode (all we have today that we can use).
  // We could sniff the file ID and switch off to other module types.
  // The module takes ownership of the file contents (when successful).
  iree_vm_module_t* module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      instance, file_contents->const_buffer,
      iree_file_contents_deallocator(file_contents), host_allocator, &module);

  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    iree_file_contents_free(file_contents);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_load_dynamic_module(
    iree_vm_instance_t* instance, iree_string_view_t path,
    iree_string_view_t export_name, iree_string_view_t params,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, params.data, params.size);

  // Split up params list - comes in `key=value&key=value` form.
  iree_host_size_t param_count = 0;
  iree_uri_split_params(params, 0, &param_count, NULL);
  iree_string_pair_t* param_list = NULL;
  if (param_count > 0) {
    param_list =
        (iree_string_pair_t*)iree_alloca(param_count * sizeof(*param_list));
    iree_uri_split_params(params, param_count, &param_count, param_list);
  }

  iree_status_t status = iree_vm_dynamic_module_load_from_file(
      instance, path, export_name, param_count, param_list, host_allocator,
      out_module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_load_modules_from_flags(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_tooling_module_list_t* list) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(list);
  iree_host_size_t new_count = list->count + FLAG_module_list().count;
  if (new_count > list->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many modules; currently only %zu are "
                            "supported but at least %zu are requested",
                            list->capacity, new_count);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < FLAG_module_list().count; ++i) {
    // We support `file.so@export_name?params` syntax to allow loading multiple
    // modules from the same shared library and passing in parameters. When
    // omitted we'll use the default export name.
    iree_string_view_t path, export_name, params;
    iree_string_view_split(FLAG_module_list().values[i], '@', &path,
                           &export_name);
    iree_string_view_split(export_name, '?', &export_name, &params);

    // Load the module based on its (guessed) type.
    iree_vm_module_t* module = NULL;
    if (iree_file_path_is_dynamic_library(path)) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_tooling_load_dynamic_module(instance, path, export_name, params,
                                           host_allocator, &module),
          "loading dynamic module at '%.*s'", (int)path.size, path.data);
    } else {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_tooling_load_bytecode_module(instance, path, host_allocator,
                                            &module),
          "loading bytecode module at '%.*s'", (int)path.size, path.data);
    }

    // Store loaded module in the list. It'll be the caller's responsibility to
    // clean it up even if we fail while loading more.
    list->values[list->count++] = module;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// HAL execution model management
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_load_hal_async_module(
    iree_vm_instance_t* instance, iree_string_view_t default_device_uri,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_ASSERT_ARGUMENT(out_device_allocator);
  if (*out_device || *out_device_allocator) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "async HAL module can not be used with other primary HAL module types");
  }
  *out_module = NULL;
  *out_device = NULL;
  *out_device_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Register required types before creating the module.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_module_register_all_types(instance));

  // Create the device to use.
  // In the future this will change to a set of available devices instead.
  if (iree_string_view_is_empty(default_device_uri)) {
    default_device_uri = iree_hal_default_device_uri();
  }
  iree_hal_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_create_device_from_flags(
              iree_hal_available_driver_registry(), default_device_uri,
              host_allocator, &device));

  // Fetch the allocator from the device to pass back to the caller.
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  iree_hal_allocator_retain(device_allocator);

  // Create HAL module wrapping the device created above.
  iree_hal_module_flags_t flags = IREE_HAL_MODULE_FLAG_NONE;
  iree_vm_module_t* module = NULL;
  iree_status_t status =
      iree_hal_module_create(instance, device, flags, host_allocator, &module);

  if (iree_status_is_ok(status)) {
    *out_module = module;
    *out_device = device;
    *out_device_allocator = device_allocator;
  } else {
    iree_hal_allocator_release(device_allocator);
    iree_hal_device_release(device);
    iree_vm_module_release(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates a HAL device allocator for host-local heap usage.
// TODO(benvanik): generalize to allocator wrapper flag (suballocator, etc).
static iree_status_t iree_tooling_create_inline_device_allocator_from_flags(
    iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_device_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_allocator_create_heap(
      IREE_SV("heap"), host_allocator, host_allocator, out_device_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_load_hal_inline_module(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module,
    iree_hal_allocator_t** out_device_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  IREE_ASSERT_ARGUMENT(out_device_allocator);
  if (*out_device_allocator) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "inline HAL module cannot be used with other "
                            "primary HAL module types");
  }
  *out_module = NULL;
  *out_device_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Register required types before creating the module.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_module_register_inline_types(instance));

  // Create default heap device allocator.
  iree_hal_allocator_t* device_allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_create_inline_device_allocator_from_flags(
              host_allocator, &device_allocator));

  // Create the module; it's immutable and can be reused but we don't do that in
  // this tooling.
  iree_hal_inline_module_flags_t flags = IREE_HAL_INLINE_MODULE_FLAG_NONE;
  iree_vm_module_t* module = NULL;
  iree_status_t status = iree_hal_inline_module_create(
      instance, flags, device_allocator, host_allocator, &module);

  if (iree_status_is_ok(status)) {
    *out_module = module;
    *out_device_allocator = device_allocator;
  } else {
    iree_hal_allocator_release(device_allocator);
    iree_vm_module_release(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_load_hal_loader_module(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Register required types before creating the module.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_module_register_loader_types(instance));

  // Create plugin manager for executable imports.
  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_plugin_manager_create_from_flags(
              host_allocator, &plugin_manager));

  // Create all executable loaders built into the binary.
  // We could allow users to choose the set with a flag.
  iree_host_size_t loader_count = 0;
  iree_hal_executable_loader_t* loaders[16];
  iree_status_t status = iree_hal_create_all_available_executable_loaders(
      plugin_manager, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
      host_allocator);

  // Create the module; it retains the loaders for its lifetime.
  iree_vm_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_loader_module_flags_t flags = IREE_HAL_LOADER_MODULE_FLAG_NONE;
    status = iree_hal_loader_module_create(instance, flags, loader_count,
                                           loaders, host_allocator, &module);
  }

  // Always release loaders; loader module has retained them.
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_hal_executable_plugin_manager_release(plugin_manager);

  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    iree_vm_module_release(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

void iree_tooling_module_list_initialize(iree_tooling_module_list_t* out_list) {
  out_list->capacity = IREE_ARRAYSIZE(out_list->values);
  out_list->count = 0;
  memset(out_list->values, 0, sizeof(out_list->values));
}

void iree_tooling_module_list_clone(
    const iree_tooling_module_list_t* source_list,
    iree_tooling_module_list_t* out_list) {
  iree_tooling_module_list_initialize(out_list);
  for (iree_host_size_t i = 0; i < source_list->count; ++i) {
    iree_vm_module_t* module = source_list->values[i];
    iree_vm_module_retain(module);
    out_list->values[out_list->count++] = module;
  }
}

void iree_tooling_module_list_reset(iree_tooling_module_list_t* list) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_vm_module_release(list->values[i]);
  }
  list->count = 0;
}

// Returns true if |list| contains a module with the given |module_name|.
static bool iree_tooling_module_list_contains(
    const iree_tooling_module_list_t* list, iree_string_view_t module_name) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (iree_string_view_equal(iree_vm_module_name(list->values[i]),
                               module_name)) {
      return true;
    }
  }
  return false;
}

iree_status_t iree_tooling_module_list_push_back(
    iree_tooling_module_list_t* list, iree_vm_module_t* module) {
  if (list->count + 1 > list->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "resolved module list capacity %" PRIhsz
                            " too small to fit all resolved modules",
                            list->capacity);
  }
  iree_vm_module_retain(module);
  list->values[list->count++] = module;
  return iree_ok_status();
}

iree_vm_module_t* iree_tooling_module_list_back(
    const iree_tooling_module_list_t* list) {
  return list->count ? list->values[list->count - 1] : NULL;
}

typedef struct {
  iree_vm_instance_t* instance;
  iree_allocator_t host_allocator;
  iree_tooling_module_list_t* resolved_list;
  iree_string_view_t default_device_uri;
  iree_hal_device_t* device;
  iree_hal_allocator_t* device_allocator;
} iree_tooling_resolve_state_t;
static iree_status_t iree_tooling_resolve_module_dependency(
    void* user_data_ptr, const iree_vm_module_dependency_t* dependency) {
  iree_tooling_resolve_state_t* state =
      (iree_tooling_resolve_state_t*)user_data_ptr;
  if (iree_tooling_module_list_contains(state->resolved_list,
                                        dependency->name)) {
    // Already registered (redundant system dep or another user module).
    return iree_ok_status();
  }

  // Register one of the known modules. If we had a factory mechanism for
  // resolving the modules we'd call out to that. Note that today this is not
  // recursive but it could be in the future.
  iree_vm_module_t* module = NULL;
  if (iree_string_view_equal(dependency->name, IREE_SV("hal"))) {
    IREE_RETURN_IF_ERROR(iree_tooling_load_hal_async_module(
        state->instance, state->default_device_uri, state->host_allocator,
        &module, &state->device, &state->device_allocator));
  } else if (iree_string_view_equal(dependency->name, IREE_SV("hal_inline"))) {
    IREE_RETURN_IF_ERROR(iree_tooling_load_hal_inline_module(
        state->instance, state->host_allocator, &module,
        &state->device_allocator));
  } else if (iree_string_view_equal(dependency->name, IREE_SV("hal_loader"))) {
    IREE_RETURN_IF_ERROR(iree_tooling_load_hal_loader_module(
        state->instance, state->host_allocator, &module));
  } else if (iree_string_view_equal(dependency->name, IREE_SV("vmvx"))) {
    IREE_RETURN_IF_ERROR(iree_vmvx_module_create(
        state->instance, state->host_allocator, &module));
  } else if (iree_all_bits_set(dependency->flags,
                               IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED)) {
    // Required but not found; fail.
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "required module '%.*s' not registered on the context",
        (int)dependency->name.size, dependency->name.data);
  } else {
    // Optional and not found; skip.
    return iree_ok_status();
  }

  iree_status_t status =
      iree_tooling_module_list_push_back(state->resolved_list, module);
  iree_vm_module_release(module);
  return status;
}

// NOTE: today we don't have fancy resolution and only check for specific system
// modules. A dynamic resolution utility that made callbacks for each module
// would be really handy to have in the base API.
//
// With our current single-pass approach we just scan the user modules to find
// the system deps and then append the user modules. We could instead do a
// recursive scan to build a potential set list, coalesce to find the minimum
// required versions, and then call a user callback to load runtime-registered
// modules.
iree_status_t iree_tooling_resolve_modules(
    iree_vm_instance_t* instance, iree_host_size_t user_module_count,
    iree_vm_module_t** user_modules, iree_string_view_t default_device_uri,
    iree_allocator_t host_allocator, iree_tooling_module_list_t* resolved_list,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(!user_module_count || user_modules);
  IREE_ASSERT_ARGUMENT(resolved_list);
  if (out_device) *out_device = NULL;
  if (out_device_allocator) *out_device_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tooling_resolve_state_t resolve_state = {
      .instance = instance,
      .host_allocator = host_allocator,
      .resolved_list = resolved_list,
      .default_device_uri = default_device_uri,
      .device = NULL,
      .device_allocator = NULL,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < user_module_count; ++i) {
    iree_vm_module_t* user_module = user_modules[i];
    status = iree_vm_module_enumerate_dependencies(
        user_module, iree_tooling_resolve_module_dependency, &resolve_state);
    if (!iree_status_is_ok(status)) {
      iree_string_view_t module_name = iree_vm_module_name(user_module);
      (void)module_name;
      status =
          iree_status_annotate_f(status, "resolving dependencies for '%.*s'",
                                 (int)module_name.size, module_name.data);
      break;
    }
    status = iree_tooling_module_list_push_back(resolved_list, user_modules[i]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    if (out_device_allocator) {
      *out_device_allocator = resolve_state.device_allocator;
    } else {
      iree_hal_allocator_release(resolve_state.device_allocator);
    }
    if (out_device) {
      *out_device = resolve_state.device;
    } else {
      iree_hal_device_release(resolve_state.device);
    }
  } else {
    iree_hal_allocator_release(resolve_state.device_allocator);
    iree_hal_device_release(resolve_state.device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_find_single_exported_function(
    iree_vm_module_t* module, iree_vm_function_t* out_function) {
  memset(out_function, 0, sizeof(*out_function));
  iree_vm_module_signature_t module_signature =
      iree_vm_module_signature(module);
  iree_host_size_t exported_functions = 0;
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    iree_vm_function_t function = {0};
    IREE_RETURN_IF_ERROR(
        iree_vm_module_lookup_function_by_ordinal(
            module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function),
        "looking up function export %zu", i);
    iree_string_view_t function_name = iree_vm_function_name(&function);
    if (iree_string_view_starts_with(function_name,
                                     iree_make_cstring_view("__")) ||
        iree_string_view_find_char(function_name, '$', 0) !=
            IREE_STRING_VIEW_NPOS) {
      // Function was either internal or special; we don't want to run these
      // as they have special ABI requirements or must only be called in
      // specific situations (module initializers, etc).
      continue;
    }
    if (exported_functions == 0) *out_function = function;
    ++exported_functions;
  }
  if (exported_functions == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "no exported functions found in module; at least one must be present");
  } else if (exported_functions > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "more than one exported function present; "
                            "--function= must be specified explicitly");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, trace_execution, false, "Traces VM execution to stderr.");

iree_status_t iree_tooling_create_instance(iree_allocator_t host_allocator,
                                           iree_vm_instance_t** out_instance) {
  IREE_ASSERT_ARGUMENT(out_instance);
  *out_instance = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT, host_allocator,
                                  &instance));

  // HACK: to load modules we need the types registered even though we don't
  // know if the types are used.
  iree_status_t status = iree_hal_module_register_all_types(instance);

  if (iree_status_is_ok(status)) {
    *out_instance = instance;
  } else {
    iree_vm_instance_release(instance);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_create_context_from_flags(
    iree_vm_instance_t* instance, iree_host_size_t user_module_count,
    iree_vm_module_t** user_modules, iree_string_view_t default_device_uri,
    iree_allocator_t host_allocator, iree_vm_context_t** out_context,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(!user_module_count || user_modules);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  if (out_device) *out_device = NULL;
  if (out_device_allocator) *out_device_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve all module dependencies into an ordered list.
  // All modules are retained in the list.
  iree_tooling_module_list_t resolved_list;
  iree_tooling_module_list_initialize(&resolved_list);
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_resolve_modules(
              instance, user_module_count, user_modules, default_device_uri,
              host_allocator, &resolved_list, &device, &device_allocator));

  iree_vm_context_flags_t flags = IREE_VM_CONTEXT_FLAG_NONE;
  if (FLAG_trace_execution) {
    // This enables tracing for all invocations but even if not set each
    // invocation can have the flag specified to trace.
    flags |= IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION;
  }

  // Create the context with the full list of resolved modules.
  // The context retains the modules and we can release them afterward.
  iree_vm_context_t* context = NULL;
  iree_status_t status = iree_vm_context_create_with_modules(
      instance, flags, resolved_list.count, resolved_list.values,
      host_allocator, &context);
  iree_tooling_module_list_reset(&resolved_list);

  // If no device allocator was created we'll create a default one just so that
  // callers have something to create buffer views from. This isn't strictly
  // required but a lot of tests do things like pass in buffers even if no HAL
  // methods are used and the HAL module is not needed.
  if (iree_status_is_ok(status) && !device_allocator && out_device_allocator) {
    status = iree_tooling_create_inline_device_allocator_from_flags(
        host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_context = context;
    if (out_device_allocator) {
      *out_device_allocator = device_allocator;
    } else {
      iree_hal_allocator_release(device_allocator);
    }
    if (out_device) {
      *out_device = device;
    } else {
      iree_hal_device_release(device);
    }
  } else {
    iree_hal_allocator_release(device_allocator);
    iree_hal_device_release(device);
    iree_vm_context_release(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
