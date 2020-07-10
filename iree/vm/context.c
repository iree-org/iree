// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/vm/context.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "iree/base/atomics.h"
#include "iree/base/tracing.h"

struct iree_vm_context {
  iree_atomic_intptr_t ref_count;
  iree_vm_instance_t* instance;
  iree_allocator_t allocator;
  intptr_t context_id;

  bool is_static;
  struct {
    iree_host_size_t count;
    iree_host_size_t capacity;
    iree_vm_module_t** modules;
    iree_vm_module_state_t** module_states;
  } list;
};

static void iree_vm_context_destroy(iree_vm_context_t* context);

// Runs a single `() -> ()` function from the module if it exists.
static iree_status_t iree_vm_context_run_function(
    iree_vm_stack_t* stack, iree_vm_module_t* module,
    iree_string_view_t function_name) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  iree_status_t status = iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name, &call.function);
  if (iree_status_is_not_found(status)) {
    // Function doesn't exist; that's ok as this was an optional call.
    IREE_TRACE_ZONE_END(z0);
    return IREE_STATUS_OK;
  } else if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_vm_execution_result_t result;
  status = module->begin_call(module->self, stack, &call, &result);

  // TODO(benvanik): ensure completed synchronously.

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_vm_context_query_module_state(
    void* state_resolver, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
  if (!state_resolver) return IREE_STATUS_INVALID_ARGUMENT;
  iree_vm_context_t* context = (iree_vm_context_t*)state_resolver;
  // NOTE: this is a linear scan, but given that the list of modules should be
  // N<4 this is faster than just about anything else we could do.
  // To future performance profilers: sorry when N>>4 :)
  for (int i = 0; i < context->list.count; ++i) {
    if (context->list.modules[i] == module) {
      *out_module_state = context->list.module_states[i];
      return IREE_STATUS_OK;
    }
  }
  return IREE_STATUS_NOT_FOUND;
}

static iree_status_t iree_vm_context_resolve_module_imports(
    iree_vm_context_t* context, iree_vm_module_t* module,
    iree_vm_module_state_t* module_state) {
  // NOTE: this has some bad characteristics, but the number of modules and the
  // number of imported functions should be relatively small (even if the number
  // of exported functions for particular modules is large).
  iree_vm_module_signature_t module_signature = module->signature(module->self);
  for (int i = 0; i < module_signature.import_function_count; ++i) {
    iree_string_view_t full_name;
    IREE_RETURN_IF_ERROR(
        module->get_function(module->self, IREE_VM_FUNCTION_LINKAGE_IMPORT, i,
                             /*out_function=*/NULL,
                             /*out_name=*/&full_name,
                             /*out_signature=*/NULL));
    iree_vm_function_t import_function;
    iree_status_t status =
        iree_vm_context_resolve_function(context, full_name, &import_function);
    if (!iree_status_is_ok(status)) {
      // iree_string_view_t is not NUL-terminated, so this is a bit awkward.
      fprintf(stderr, "Unable to resolve VM function '");
      for (size_t i = 0; i < full_name.size; i++) {
        fputc(full_name.data[i], stderr);
      }
      fprintf(stderr, "'. Ensure modules are registered with the context.\n");
      return status;
    }
    IREE_RETURN_IF_ERROR(
        module->resolve_import(module->self, module_state, i, import_function));
  }
  return IREE_STATUS_OK;
}

static void iree_vm_context_release_modules(iree_vm_context_t* context,
                                            iree_host_size_t start,
                                            iree_host_size_t end) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Run module __deinit functions, if present (in reverse init order).
  IREE_VM_INLINE_STACK_INITIALIZE(
      stack, iree_vm_context_state_resolver(context), context->allocator);
  for (int i = (int)end; i >= (int)start; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    iree_vm_module_state_t* module_state = context->list.module_states[i];
    if (!module_state) {
      // Partially initialized; skip.
      continue;
    }
    IREE_IGNORE_ERROR(iree_vm_context_run_function(
        stack, module, iree_make_cstring_view("__deinit")));
  }
  iree_vm_stack_deinitialize(stack);

  // Release all module state (in reverse init order).
  for (int i = (int)end; i >= (int)start; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    // It is possible in error states to have partially initialized.
    if (context->list.module_states[i]) {
      module->free_state(module->self, context->list.module_states[i]);
    }
    context->list.module_states[i] = NULL;
  }

  // Release modules now that there are no import tables remaining.
  for (int i = (int)end; i >= (int)start; --i) {
    if (context->list.modules[i]) {
      iree_vm_module_release(context->list.modules[i]);
    }
    context->list.modules[i] = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_context_create(iree_vm_instance_t* instance, iree_allocator_t allocator,
                       iree_vm_context_t** out_context) {
  return iree_vm_context_create_with_modules(instance, NULL, 0, allocator,
                                             out_context);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_context_create_with_modules(
    iree_vm_instance_t* instance, iree_vm_module_t** modules,
    iree_host_size_t module_count, iree_allocator_t allocator,
    iree_vm_context_t** out_context) {
  if (!out_context) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_context = NULL;

  if (!instance) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (!modules && module_count > 0) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  for (int i = 0; i < module_count; ++i) {
    if (!modules[i]) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
  }

  iree_host_size_t context_size =
      sizeof(iree_vm_context_t) + sizeof(iree_vm_module_t*) * module_count +
      sizeof(iree_vm_module_state_t*) * module_count;

  iree_vm_context_t* context = NULL;
  iree_allocator_malloc(allocator, context_size, (void**)&context);
  iree_atomic_store(&context->ref_count, 1);
  context->instance = instance;
  iree_vm_instance_retain(context->instance);
  context->allocator = allocator;

  static iree_atomic_intptr_t next_context_id = IREE_ATOMIC_VAR_INIT(1);
  context->context_id = iree_atomic_fetch_add(&next_context_id, 1);

  uint8_t* p = (uint8_t*)context + sizeof(iree_vm_context_t);
  context->list.modules = (iree_vm_module_t**)p;
  p += sizeof(iree_vm_module_t*) * module_count;
  context->list.module_states = (iree_vm_module_state_t**)p;
  p += sizeof(iree_vm_module_state_t*) * module_count;
  context->list.count = 0;
  context->list.capacity = module_count;
  context->is_static = module_count > 0;

  iree_status_t register_status =
      iree_vm_context_register_modules(context, modules, module_count);
  if (!iree_status_is_ok(register_status)) {
    iree_vm_context_destroy(context);
    return register_status;
  }

  *out_context = context;
  return IREE_STATUS_OK;
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

IREE_API_EXPORT void IREE_API_CALL
iree_vm_context_retain(iree_vm_context_t* context) {
  if (context) {
    iree_atomic_fetch_add(&context->ref_count, 1);
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_context_release(iree_vm_context_t* context) {
  if (context && iree_atomic_fetch_sub(&context->ref_count, 1) == 1) {
    iree_vm_context_destroy(context);
  }
}

IREE_API_EXPORT intptr_t IREE_API_CALL
iree_vm_context_id(const iree_vm_context_t* context) {
  if (!context) {
    return -1;
  }
  return context->context_id;
}

IREE_API_EXPORT iree_vm_state_resolver_t IREE_API_CALL
iree_vm_context_state_resolver(const iree_vm_context_t* context) {
  iree_vm_state_resolver_t state_resolver = {0};
  state_resolver.self = (void*)context;
  state_resolver.query_module_state = iree_vm_context_query_module_state;
  return state_resolver;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_context_resolve_module_state(
    const iree_vm_context_t* context, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
  return iree_vm_context_query_module_state((void*)context, module,
                                            out_module_state);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_context_register_modules(
    iree_vm_context_t* context, iree_vm_module_t** modules,
    iree_host_size_t module_count) {
  if (!context) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (!modules && module_count > 1) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  for (int i = 0; i < module_count; ++i) {
    if (!modules[i]) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
  }

  // Try growing both our storage lists first, if needed.
  if (context->list.count + module_count > context->list.capacity) {
    if (context->is_static) {
      return IREE_STATUS_FAILED_PRECONDITION;
    }
    iree_host_size_t new_capacity = context->list.capacity + module_count;
    if (new_capacity < context->list.capacity * 2) {
      // TODO(benvanik): tune list growth for module count >> 4.
      new_capacity = context->list.capacity * 2;
    }
    iree_vm_module_t** new_module_list;
    iree_allocator_malloc(context->allocator,
                          sizeof(iree_vm_module_t*) * new_capacity,
                          (void**)&new_module_list);
    iree_vm_module_state_t** new_module_state_list;
    iree_allocator_malloc(context->allocator,
                          sizeof(iree_vm_module_state_t*) * new_capacity,
                          (void**)&new_module_state_list);
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
      stack, iree_vm_context_state_resolver(context), context->allocator);

  // Retain all modules and allocate their state.
  assert(context->list.capacity >= context->list.count + module_count);
  iree_host_size_t original_count = context->list.count;
  iree_status_t status = IREE_STATUS_OK;
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
      break;
    }

    ++context->list.count;

    // Run module __init functions, if present.
    // As initialization functions may reference imports we need to perform
    // all of these after we have resolved the imports above.
    status = iree_vm_context_run_function(stack, module,
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

  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_context_resolve_function(
    const iree_vm_context_t* context, iree_string_view_t full_name,
    iree_vm_function_t* out_function) {
  if (!out_function) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  memset(out_function, 0, sizeof(iree_vm_function_t));

  iree_string_view_t module_name;
  iree_string_view_t function_name;
  if (iree_string_view_split(full_name, '.', &module_name, &function_name) ==
      -1) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  for (int i = (int)context->list.count - 1; i >= 0; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    if (iree_string_view_compare(module_name, iree_vm_module_name(module)) ==
        0) {
      return iree_vm_module_lookup_function_by_name(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name, out_function);
    }
  }

  return IREE_STATUS_NOT_FOUND;
}
