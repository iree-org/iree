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

#include "iree/vm2/context.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdbool.h>

struct iree_vm_context {
  atomic_intptr_t ref_count;
  iree_vm_instance_t* instance;
  iree_allocator_t allocator;
  int32_t context_id;

  bool is_static;
  struct {
    iree_host_size_t count;
    iree_host_size_t capacity;
    iree_vm_module_t** modules;
    iree_vm_module_state_t** module_states;
  } list;
};

static iree_status_t iree_vm_context_destroy(iree_vm_context_t* context);

static iree_status_t iree_vm_context_query_module_state(
    void* state_resolver, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
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
    IREE_API_RETURN_IF_API_ERROR(
        module->get_function(module->self, IREE_VM_FUNCTION_LINKAGE_IMPORT, i,
                             /*out_function=*/NULL,
                             /*out_name=*/&full_name,
                             /*out_signature=*/NULL));
    iree_vm_function_t import_function;
    IREE_API_RETURN_IF_API_ERROR(
        iree_vm_context_resolve_function(context, full_name, &import_function));
    IREE_API_RETURN_IF_API_ERROR(
        module->resolve_import(module->self, module_state, i, import_function));
  }
  return IREE_STATUS_OK;
}

static void iree_vm_context_release_modules(iree_vm_context_t* context,
                                            int start, int end) {
  for (int i = end; i >= start; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    // It is possible in error states to have partially initialized.
    if (context->list.module_states[i]) {
      module->free_state(module->self, context->list.module_states[i]);
    }
    context->list.module_states[i] = NULL;
  }
  for (int i = end; i >= start; --i) {
    if (context->list.modules[i]) {
      iree_vm_module_release(context->list.modules[i]);
    }
    context->list.modules[i] = NULL;
  }
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
  IREE_API_RETURN_IF_API_ERROR(
      iree_allocator_malloc(allocator, context_size, (void**)&context));
  atomic_store(&context->ref_count, 1);
  context->instance = instance;
  iree_vm_instance_retain(context->instance);
  context->allocator = allocator;

  static atomic_int next_context_id = 0;
  context->context_id = atomic_fetch_add(&next_context_id, 1);

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
  if (register_status != IREE_STATUS_OK) {
    iree_vm_context_destroy(context);
    return register_status;
  }

  *out_context = context;
  return IREE_STATUS_OK;
}

static iree_status_t iree_vm_context_destroy(iree_vm_context_t* context) {
  if (!context) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

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
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_context_retain(iree_vm_context_t* context) {
  if (!context) return IREE_STATUS_INVALID_ARGUMENT;
  atomic_fetch_add(&context->ref_count, 1);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_context_release(iree_vm_context_t* context) {
  if (context) {
    if (atomic_fetch_sub(&context->ref_count, 1) == 1) {
      return iree_vm_context_destroy(context);
    }
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT int32_t IREE_API_CALL
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
    // The existing memory is only dynamically allocated if it has been grown.
    if (context->list.capacity > 0) {
      iree_allocator_free(context->allocator, context->list.modules);
      iree_allocator_free(context->allocator, context->list.module_states);
    }
    context->list.modules = new_module_list;
    context->list.module_states = new_module_state_list;
    context->list.capacity = new_capacity;
  }

  // Retain all modules and allocate their state.
  assert(context->list.capacity >= context->list.count + module_count);
  int orig_count = context->list.count;
  for (int i = 0; i < module_count; ++i) {
    iree_vm_module_t* module = modules[i];
    context->list.modules[orig_count + i] = module;
    context->list.module_states[orig_count + i] = NULL;

    iree_vm_module_retain(module);

    // Allocate module state.
    iree_vm_module_state_t* module_state = NULL;
    iree_status_t alloc_status =
        module->alloc_state(module->self, context->allocator, &module_state);
    if (alloc_status != IREE_STATUS_OK) {
      // NOTE: we need to clean up initialized modules.
      iree_vm_context_release_modules(context, orig_count, orig_count + i);
      context->list.count = orig_count;
      return alloc_status;
    }
    context->list.module_states[orig_count + i] = module_state;

    // Resolve imports for the modules.
    // TODO(benvanik): re-resolve imports for previous modules?
    iree_status_t resolve_status =
        iree_vm_context_resolve_module_imports(context, module, module_state);
    if (resolve_status != IREE_STATUS_OK) {
      // NOTE: we need to clean up initialized modules.
      iree_vm_context_release_modules(context, orig_count, orig_count + i);
      context->list.count = orig_count;
      return resolve_status;
    }

    ++context->list.count;
  }

  return IREE_STATUS_OK;
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

  for (int i = context->list.count - 1; i >= 0; --i) {
    iree_vm_module_t* module = context->list.modules[i];
    if (iree_string_view_compare(module_name, iree_vm_module_name(module)) ==
        0) {
      return iree_vm_module_lookup_function(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name, out_function);
    }
  }

  return IREE_STATUS_NOT_FOUND;
}
