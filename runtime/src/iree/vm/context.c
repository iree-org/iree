// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/context.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"
#include "iree/base/tracing.h"

struct iree_vm_context_t {
  iree_atomic_ref_count_t ref_count;
  iree_vm_instance_t* instance;
  iree_allocator_t allocator;

  // An opaque ID unique for the entire process lifetime.
  // If tracing then this points at a NUL-terminated string with process
  // lifetime.
  iree_vm_context_id_t context_id;

  // Context has been frozen and can no longer be modified.
  uint32_t is_frozen : 1;
  // Context storage is statically allocated and need not be freed.
  uint32_t is_static : 1;

  // Configuration flags.
  iree_vm_context_flags_t flags;

  struct {
    iree_host_size_t count;
    iree_host_size_t capacity;
    iree_vm_module_t** modules;
    iree_vm_module_state_t** module_states;
  } list;
};

static iree_status_t iree_vm_context_resolve_function_impl(
    const iree_vm_context_t* context, iree_string_view_t full_name,
    const iree_vm_function_signature_t* expected_signature,
    iree_vm_function_t* out_function);

static void iree_vm_context_destroy(iree_vm_context_t* context);

// Allocates a process-unique ID for a context to use.
static iree_vm_context_id_t iree_vm_context_allocate_id(void) {
  static iree_atomic_int32_t next_context_id = IREE_ATOMIC_VAR_INIT(1);
  // relaxed because we only care about atomic increments, not ordering w.r.t.
  // other memory accesses.
  uint32_t context_id = iree_atomic_fetch_add_int32(&next_context_id, 1,
                                                    iree_memory_order_relaxed);
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FIBERS
  // This is what we pass to Tracy as the fiber name.
  // The string must remain live for the lifetime of the process.
  IREE_LEAK_CHECK_DISABLE_PUSH();
  char* name = (char*)malloc(32);
  snprintf(name, 32, "ctx-%04d", context_id - 1);
  IREE_LEAK_CHECK_DISABLE_POP();
  return (iree_vm_context_id_t)name;
#else
  return (iree_vm_context_id_t)context_id;
#endif  // IREE_TRACING_FEATURE_FIBERS
}

// Runs a single `() -> ()` function from the module if it exists.
static iree_status_t iree_vm_context_run_function(
    iree_vm_context_t* context, iree_vm_stack_t* stack,
    iree_vm_module_t* module, iree_string_view_t function_name) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  iree_status_t status = iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name, &call.function);
  if (iree_status_is_not_found(status)) {
    // Function doesn't exist; that's ok as this was an optional call.
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  } else if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_FIBER_ENTER(context->context_id);
  status = module->begin_call(module->self, stack, call);
  IREE_TRACE_FIBER_LEAVE();
  if (!iree_status_is_ok(status)) {
    status = IREE_VM_STACK_ANNOTATE_BACKTRACE_IF_ENABLED(stack, status);
  }

  // TODO(benvanik): ensure completed synchronously.

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_vm_context_query_module_state(
    void* state_resolver, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
  IREE_ASSERT_ARGUMENT(state_resolver);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_module_state);
  iree_vm_context_t* context = (iree_vm_context_t*)state_resolver;
  // NOTE: this is a linear scan, but given that the list of modules should be
  // N<4 this is faster than just about anything else we could do.
  // To future performance profilers: sorry when N>>4 :)
  for (int i = 0; i < context->list.count; ++i) {
    if (context->list.modules[i] == module) {
      *out_module_state = context->list.module_states[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

// Checks that |dependency| is satisfied by the context.
static iree_status_t iree_vm_context_check_module_dependency(
    void* user_data_ptr, const iree_vm_module_dependency_t* dependency) {
  // Scan the context to find the dependency by module name.
  iree_vm_context_t* context = (iree_vm_context_t*)user_data_ptr;
  for (iree_host_size_t i = 0; i < context->list.count; ++i) {
    iree_vm_module_t* module = context->list.modules[i];
    if (iree_string_view_equal(iree_vm_module_name(module), dependency->name)) {
      iree_vm_module_signature_t signature = iree_vm_module_signature(module);
      if (iree_all_bits_set(dependency->flags,
                            IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED)) {
        if (signature.version < dependency->minimum_version) {
          // Required modules must meet the version requirement.
          return iree_make_status(
              IREE_STATUS_NOT_FOUND,
              "required module '%.*s' version mismatch; have %u but require %u",
              (int)dependency->name.size, dependency->name.data,
              signature.version, dependency->minimum_version);
        }
        // Found and version matches.
        return iree_ok_status();
      } else {
        // Found the module and it's optional so allow all versions.
        return iree_ok_status();
      }
    }
  }
  // Optional dependencies are not failures when not found.
  if (iree_all_bits_set(dependency->flags,
                        IREE_VM_MODULE_DEPENDENCY_FLAG_OPTIONAL)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "required module '%.*s' not registered on the context",
      (int)dependency->name.size, dependency->name.data);
}

static iree_status_t iree_vm_context_resolve_module_imports(
    iree_vm_context_t* context, iree_vm_module_t* module,
    iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check module presence/versions before individual imports.
  // This gives better error messages ("requires hal module version 4") than
  // failing on individual methods.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_module_enumerate_dependencies(
              module, iree_vm_context_check_module_dependency, context));

  // NOTE: this has some bad characteristics, but the number of modules and the
  // number of imported functions should be relatively small (even if the number
  // of exported functions for particular modules is large).
  iree_vm_module_signature_t module_signature = module->signature(module->self);
  for (int i = 0; i < module_signature.import_function_count; ++i) {
    iree_vm_function_t decl_function;
    iree_string_view_t full_name;
    iree_vm_function_signature_t expected_signature;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        module->get_function(module->self, IREE_VM_FUNCTION_LINKAGE_IMPORT, i,
                             /*out_function=*/&decl_function,
                             /*out_name=*/&full_name,
                             /*out_signature=*/&expected_signature));

    // Resolve the function to the module that contains it and return the
    // information.
    iree_vm_function_t import_function;
    iree_status_t resolve_status = iree_vm_context_resolve_function_impl(
        context, full_name, &expected_signature, &import_function);
    if (!iree_status_is_ok(resolve_status)) {
      if (iree_status_is_not_found(resolve_status) &&
          decl_function.linkage == IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL) {
        // Failed to find the function but it was optionally imported and that's
        // ok. We'll just continue the resolution process and leave the import
        // unspecified on the target module.
        iree_status_ignore(resolve_status);
        continue;
      } else {
        // Failed to find the function.
        IREE_TRACE_ZONE_END(z0);
        return resolve_status;
      }
    }

    // Query the function signature from the module that contains it; we don't
    // use the signature from the module requesting the import as we want a
    // single source of truth.
    iree_vm_function_signature_t import_signature =
        iree_vm_function_signature(&import_function);

    // Simple check to confirm the signatures match. We still can't trust that
    // the module using the import *actually* calls it with the right convention
    // (so this is not a safety check!), but this will catch the 99% case of a
    // signature changing out from under a module or using a module with a newer
    // signature than that provided by the imported module.
    //
    // We allow modules to not define their cconv expectation as in a lot of
    // cases where modules are all compiled into the same binary there's no
    // value in performing the verification. Runtime checks during calls will
    // fail with less awesome logging but that's the tradeoff.
    if (expected_signature.calling_convention.size &&
        !iree_string_view_equal(import_signature.calling_convention,
                                expected_signature.calling_convention)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "import function signature mismatch between %.*s "
          "and source %.*s; expected %.*s but got %.*s",
          (int)iree_vm_module_name(module).size,
          iree_vm_module_name(module).data,
          (int)iree_vm_module_name(import_function.module).size,
          iree_vm_module_name(import_function.module).data,
          (int)expected_signature.calling_convention.size,
          expected_signature.calling_convention.data,
          (int)import_signature.calling_convention.size,
          import_signature.calling_convention.data);
    }

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, module->resolve_import(module->self, module_state, i,
                                   &import_function, &import_signature));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_vm_context_release_modules(iree_vm_context_t* context,
                                            iree_host_size_t start,
                                            iree_host_size_t end) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Run module __deinit functions, if present (in reverse init order).
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack,
      context->flags & IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
          ? IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION
          : IREE_VM_INVOCATION_FLAG_NONE,
      iree_vm_context_state_resolver(context), context->allocator);
  for (int i = (int)end; i >= (int)start; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    iree_vm_module_state_t* module_state = context->list.module_states[i];
    if (!module_state) {
      // Partially initialized; skip.
      continue;
    }
    IREE_IGNORE_ERROR(iree_vm_context_run_function(
        context, stack, module, iree_make_cstring_view("__deinit")));
  }
  iree_vm_stack_deinitialize(stack);

  // Release all module state (in reverse init order).
  for (int i = (int)end; i >= (int)start; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    // It is possible in error states to have partially initialized.
    if (context->list.module_states[i]) {
      module->free_state(module->self, context->list.module_states[i]);
      context->list.module_states[i] = NULL;
    }
  }

  // Release modules now that there are no import tables remaining.
  for (int i = (int)end; i >= (int)start; --i) {
    if (context->list.modules[i]) {
      iree_vm_module_release(context->list.modules[i]);
      context->list.modules[i] = NULL;
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_vm_context_create(
    iree_vm_instance_t* instance, iree_vm_context_flags_t flags,
    iree_allocator_t allocator, iree_vm_context_t** out_context) {
  return iree_vm_context_create_with_modules(
      instance, flags, /*module_count=*/0, /*modules=*/NULL, allocator,
      out_context);
}

IREE_API_EXPORT iree_status_t iree_vm_context_create_with_modules(
    iree_vm_instance_t* instance, iree_vm_context_flags_t flags,
    iree_host_size_t module_count, iree_vm_module_t** modules,
    iree_allocator_t allocator, iree_vm_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;

  iree_host_size_t context_size =
      sizeof(iree_vm_context_t) + sizeof(iree_vm_module_t*) * module_count +
      sizeof(iree_vm_module_state_t*) * module_count;

  iree_vm_context_t* context = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, context_size, (void**)&context));
  iree_atomic_ref_count_init(&context->ref_count);
  context->instance = instance;
  iree_vm_instance_retain(context->instance);
  context->allocator = allocator;

  context->context_id = iree_vm_context_allocate_id();

  // TODO(benvanik): allow for non-frozen but static contexts.
  context->is_frozen = module_count > 0;
  context->is_static = module_count > 0;
  context->flags = flags;

  uint8_t* p = (uint8_t*)context + sizeof(iree_vm_context_t);
  context->list.modules = (iree_vm_module_t**)p;
  p += sizeof(iree_vm_module_t*) * module_count;
  context->list.module_states = (iree_vm_module_state_t**)p;
  p += sizeof(iree_vm_module_state_t*) * module_count;
  context->list.count = 0;
  context->list.capacity = module_count;

  iree_status_t register_status =
      iree_vm_context_register_modules(context, module_count, modules);
  if (!iree_status_is_ok(register_status)) {
    iree_vm_context_destroy(context);
    IREE_TRACE_ZONE_END(z0);
    return register_status;
  }

  *out_context = context;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_vm_context_destroy(iree_vm_context_t* context) {
  if (!context) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  if (context->list.count > 0) {
    iree_vm_context_release_modules(context, 0, context->list.count - 1);
  }

  // Note: For non-static module lists, it is only dynamically allocated if
  // capacity > 0.
  if (!context->is_static && context->list.capacity > 0) {
    iree_allocator_free(context->allocator, context->list.modules);
    context->list.modules = NULL;
    iree_allocator_free(context->allocator, context->list.module_states);
    context->list.module_states = NULL;
  }

  iree_vm_instance_release(context->instance);
  context->instance = NULL;

  iree_allocator_free(context->allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_vm_context_retain(iree_vm_context_t* context) {
  if (context) {
    iree_atomic_ref_count_inc(&context->ref_count);
  }
}

IREE_API_EXPORT void iree_vm_context_release(iree_vm_context_t* context) {
  if (context && iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_vm_context_destroy(context);
  }
}

IREE_API_EXPORT iree_vm_instance_t* iree_vm_context_instance(
    const iree_vm_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->instance;
}

IREE_API_EXPORT iree_vm_context_id_t
iree_vm_context_id(const iree_vm_context_t* context) {
  if (!context) return -1;
  return context->context_id;
}

IREE_API_EXPORT iree_vm_context_flags_t
iree_vm_context_flags(const iree_vm_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->flags;
}

IREE_API_EXPORT iree_host_size_t
iree_vm_context_module_count(const iree_vm_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->list.count;
}

IREE_API_EXPORT iree_vm_module_t* iree_vm_context_module_at(
    const iree_vm_context_t* context, iree_host_size_t i) {
  IREE_ASSERT_ARGUMENT(context);
  if (i >= context->list.count) return NULL;
  return context->list.modules[i];
}

IREE_API_EXPORT iree_status_t iree_vm_context_register_modules(
    iree_vm_context_t* context, iree_host_size_t module_count,
    iree_vm_module_t** modules) {
  IREE_ASSERT_ARGUMENT(context);
  if (!modules && module_count > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "modules/module_count mismatch");
  }
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    if (!modules[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%zu] is null", i);
    }
  }
  if (!module_count) return iree_ok_status();

  IREE_TRACE_ZONE_BEGIN(z0);

  // Try growing both our storage lists first, if needed.
  if (context->list.count + module_count > context->list.capacity) {
    if (context->is_frozen) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "context was allocated as static and cannot "
                              "register modules after creation");
    }
    iree_host_size_t new_capacity = context->list.capacity + module_count;
    if (new_capacity < context->list.capacity * 2) {
      // TODO(benvanik): tune list growth for module count >> 4.
      new_capacity = context->list.capacity * 2;
    }
    iree_vm_module_t** new_module_list = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(context->allocator,
                                  sizeof(iree_vm_module_t*) * new_capacity,
                                  (void**)&new_module_list));
    iree_vm_module_state_t** new_module_state_list = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_allocator_malloc(context->allocator,
                              sizeof(iree_vm_module_state_t*) * new_capacity,
                              (void**)&new_module_state_list));
    memcpy(new_module_list, context->list.modules,
           sizeof(iree_vm_module_t*) * context->list.count);
    memcpy(new_module_state_list, context->list.module_states,
           sizeof(iree_vm_module_state_t*) * context->list.count);
    // The existing memory is only dynamically allocated if it has been
    // grown.
    if (context->list.capacity > 0) {
      iree_allocator_free(context->allocator, context->list.modules);
      iree_allocator_free(context->allocator, context->list.module_states);
    }
    context->list.modules = new_module_list;
    context->list.module_states = new_module_state_list;
    context->list.capacity = new_capacity;
  }

  // VM stack used to call into module __init methods.
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack,
      context->flags & IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
          ? IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION
          : IREE_VM_INVOCATION_FLAG_NONE,
      iree_vm_context_state_resolver(context), context->allocator);

  // Retain all modules and allocate their state.
  assert(context->list.capacity >= context->list.count + module_count);
  iree_host_size_t original_count = context->list.count;
  iree_status_t status = iree_ok_status();
  iree_host_size_t i = 0;
  for (i = 0; i < module_count; ++i) {
    iree_vm_module_t* module = modules[i];
    context->list.modules[original_count + i] = module;
    context->list.module_states[original_count + i] = NULL;

    iree_vm_module_retain(module);

    // Allocate module state.
    iree_vm_module_state_t* module_state = NULL;
    status =
        module->alloc_state(module->self, context->allocator, &module_state);
    if (!iree_status_is_ok(status)) {
      // Cleanup handled below.
      break;
    }
    context->list.module_states[original_count + i] = module_state;

    // Resolve imports for the modules.
    status =
        iree_vm_context_resolve_module_imports(context, module, module_state);
    if (!iree_status_is_ok(status)) {
      // Cleanup handled below.
      iree_string_view_t module_name = iree_vm_module_name(module);
      (void)module_name;
      status = iree_status_annotate_f(status, "resolving module '%.*s' imports",
                                      (int)module_name.size, module_name.data);
      break;
    }

    ++context->list.count;

    // Run module __init functions, if present.
    // As initialization functions may reference imports we need to perform
    // all of these after we have resolved the imports above.
    status = iree_vm_context_run_function(context, stack, module,
                                          iree_make_cstring_view("__init"));
    if (!iree_status_is_ok(status)) {
      // Cleanup handled below.
      break;
    }
  }

  iree_vm_stack_deinitialize(stack);

  // Cleanup for failure cases during module initialization; we need to
  // ensure we release any modules we'd already initialized.
  if (!iree_status_is_ok(status)) {
    iree_vm_context_release_modules(context, original_count,
                                    original_count + i);
    context->list.count = original_count;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_vm_context_freeze(iree_vm_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  context->is_frozen = 1;
  return iree_ok_status();
}

IREE_API_EXPORT iree_vm_state_resolver_t
iree_vm_context_state_resolver(const iree_vm_context_t* context) {
  iree_vm_state_resolver_t state_resolver = {0};
  state_resolver.self = (void*)context;
  state_resolver.query_module_state = iree_vm_context_query_module_state;
  return state_resolver;
}

IREE_API_EXPORT iree_status_t iree_vm_context_resolve_module_state(
    const iree_vm_context_t* context, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
  return iree_vm_context_query_module_state((void*)context, module,
                                            out_module_state);
}

static iree_status_t iree_vm_context_resolve_function_impl(
    const iree_vm_context_t* context, iree_string_view_t full_name,
    const iree_vm_function_signature_t* expected_signature,
    iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_function);
  memset(out_function, 0, sizeof(iree_vm_function_t));

  iree_string_view_t module_name;
  iree_string_view_t function_name;
  if (iree_string_view_split(full_name, '.', &module_name, &function_name) ==
      -1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "import name not fully-qualified (module.func): '%.*s'",
        (int)full_name.size, full_name.data);
  }

  for (int i = (int)context->list.count - 1; i >= 0; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    if (iree_string_view_equal(module_name, iree_vm_module_name(module))) {
      return module->lookup_function(
          module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name,
          expected_signature, out_function);
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "module '%.*s' required for import '%.*s' not "
                          "registered with the context",
                          (int)module_name.size, module_name.data,
                          (int)full_name.size, full_name.data);
}

IREE_API_EXPORT iree_status_t iree_vm_context_resolve_function(
    const iree_vm_context_t* context, iree_string_view_t full_name,
    iree_vm_function_t* out_function) {
  return iree_vm_context_resolve_function_impl(
      context, full_name, /*expected_signature=*/NULL, out_function);
}

// Calls the '__notify(i32)' function in |module|, if present.
static iree_status_t iree_vm_context_call_module_notify(
    iree_vm_stack_t* stack, iree_vm_module_t* module,
    iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  // Single i32 argument with the signal number.
  uint32_t signal_arg = (uint32_t)signal;
  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.arguments = iree_make_byte_span(&signal_arg, sizeof(signal_arg));

  // Try to find the function. Modules are not required to export it.
  iree_status_t status = iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("__notify"), &call.function);
  if (iree_status_is_not_found(status)) {
    // Function doesn't exist; that's ok as this was an optional call.
    return iree_status_ignore(status);
  } else if (!iree_status_is_ok(status)) {
    // Failed during trim.
    return status;
  }

  // Call the resolved function.
  status = module->begin_call(module->self, stack, call);
  if (!iree_status_is_ok(status)) {
    status = IREE_VM_STACK_ANNOTATE_BACKTRACE_IF_ENABLED(stack, status);
  }

  // TODO(benvanik): ensure completed synchronously.

  return status;
}

// Calls the module notify methods in registration order.
static iree_status_t iree_vm_context_notify_forward(iree_vm_stack_t* stack,
                                                    iree_vm_context_t* context,
                                                    iree_vm_signal_t signal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->list.count; ++i) {
    iree_vm_module_t* module = context->list.modules[i];
    iree_vm_module_state_t* module_state = context->list.module_states[i];

    // Call the module internal interface notify method.
    // This handles the resources owned by the module implementation itself
    // such as JITed binaries or other module infrastructure.
    status = module->notify(module->self, module_state, signal);
    if (!iree_status_is_ok(status)) break;

    // Call the user-level notify method.
    // This may new use the reallocated resources from the module internal
    // implementation above.
    status =
        iree_vm_context_call_module_notify(stack, module, module_state, signal);
    if (!iree_status_is_ok(status)) break;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Calls the module notify methods in reverse registration order.
static iree_status_t iree_vm_context_notify_reverse(iree_vm_stack_t* stack,
                                                    iree_vm_context_t* context,
                                                    iree_vm_signal_t signal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  for (int i = (int)context->list.count - 1; i >= 0; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    iree_vm_module_state_t* module_state = context->list.module_states[i];

    // Call the user-level notify method first.
    // This allows users to drop any state that they can rematerialize and
    // return the resources to pools/caches to be trimmed below.
    status =
        iree_vm_context_call_module_notify(stack, module, module_state, signal);
    if (!iree_status_is_ok(status)) break;

    // Call the module internal interface notify method.
    // This handles the resources owned by the module implementation itself
    // such as JITed binaries or other module infrastructure. Since we've
    // already called the user-level function we likely have all of the
    // resources that could be returned to pools there for this to reclaim.
    status = module->notify(module->self, module_state, signal);
    if (!iree_status_is_ok(status)) break;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_context_notify(iree_vm_context_t* context,
                                                     iree_vm_signal_t signal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)signal);

  // VM stack used to call into module __init methods.
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack,
      context->flags & IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
          ? IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION
          : IREE_VM_INVOCATION_FLAG_NONE,
      iree_vm_context_state_resolver(context), context->allocator);

  // Resumes are walked forward while suspends are walked backward.
  // This follows the expected construction/destruction pattern where for
  // example on suspend one would walk user modules to release resources back
  // to system module pools before the system modules then clean up the pools.
  iree_status_t status = iree_ok_status();
  switch (signal) {
    default:
    case IREE_VM_SIGNAL_RESUME:
      status = iree_vm_context_notify_forward(stack, context, signal);
      break;
    case IREE_VM_SIGNAL_SUSPEND:
    case IREE_VM_SIGNAL_LOW_MEMORY:
      status = iree_vm_context_notify_reverse(stack, context, signal);
      break;
  }

  iree_vm_stack_deinitialize(stack);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
