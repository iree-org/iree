// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/native_module.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/vm/stack.h"

// Native module implementation allocated for all modules.
typedef struct iree_vm_native_module_t {
  // Interface containing default function pointers.
  // base_interface.self will be the self pointer to iree_vm_native_module_t.
  //
  // Must be first in the struct as we dereference the interface to find our
  // members below.
  iree_vm_module_t base_interface;

  // Interface with optional user-provided function pointers.
  iree_vm_module_t user_interface;

  // The self passed to user_interface functions. Will either be the value of
  // user_interface.self when initialized and the base pointer of the base
  // native module otherwise.
  void* self;

  // Allocator this module was allocated with and must be freed with.
  iree_allocator_t allocator;

  // Module descriptor used for reflection.
  const iree_vm_native_module_descriptor_t* descriptor;
} iree_vm_native_module_t;

IREE_API_EXPORT iree_host_size_t iree_vm_native_module_size(void) {
  return sizeof(iree_vm_native_module_t);
}

#if defined(NDEBUG)
static iree_status_t iree_vm_native_module_verify_descriptor(
    const iree_vm_native_module_descriptor_t* module_descriptor) {
  return iree_ok_status();
}
#else
static iree_status_t iree_vm_native_module_verify_descriptor(
    const iree_vm_native_module_descriptor_t* module_descriptor) {
  // Verify the export table is sorted by name. This will help catch issues with
  // people appending to tables instead of inserting in the proper order.
  for (iree_host_size_t i = 1; i < module_descriptor->export_count; ++i) {
    iree_string_view_t prev_export_name =
        module_descriptor->exports[i - 1].local_name;
    iree_string_view_t export_name = module_descriptor->exports[i].local_name;
    int cmp = iree_string_view_compare(prev_export_name, export_name);
    if (IREE_UNLIKELY(cmp >= 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "module export table is not sorted by name "
                              "(export %zu ('%.*s') >= %zu ('%.*s'))",
                              i - 1, (int)prev_export_name.size,
                              prev_export_name.data, i, (int)export_name.size,
                              export_name.data);
    }
  }
  return iree_ok_status();
}
#endif  // NDEBUG

static void IREE_API_PTR iree_vm_native_module_destroy(void* self) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  iree_allocator_t allocator = module->allocator;

  // Destroy the optional user-provided self.
  if (module->user_interface.destroy) {
    module->user_interface.destroy(module->self);
  }

  iree_allocator_free(allocator, module);
}

static iree_string_view_t IREE_API_PTR iree_vm_native_module_name(void* self) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.name) {
    return module->user_interface.name(module->self);
  }
  return module->descriptor->name;
}

static iree_vm_module_signature_t IREE_API_PTR
iree_vm_native_module_signature(void* self) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.signature) {
    return module->user_interface.signature(module->self);
  }
  iree_vm_module_signature_t signature = {
      .version = module->descriptor->version,
      .attr_count = module->descriptor->attr_count,
      .import_function_count = module->descriptor->import_count,
      .export_function_count = module->descriptor->export_count,
      .internal_function_count = 0,  // unused
  };
  return signature;
}

static iree_status_t IREE_API_PTR iree_vm_native_module_get_module_attr(
    void* self, iree_host_size_t index, iree_string_pair_t* out_attr) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.get_module_attr) {
    return module->user_interface.get_module_attr(module->self, index,
                                                  out_attr);
  } else if (index >= module->descriptor->attr_count) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  *out_attr = module->descriptor->attrs[index];
  return iree_ok_status();
}

static iree_status_t iree_vm_native_module_enumerate_dependencies(
    void* self, iree_vm_module_dependency_callback_t callback,
    void* user_data) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.enumerate_dependencies) {
    return module->user_interface.enumerate_dependencies(module->self, callback,
                                                         user_data);
  }
  for (iree_host_size_t i = 0; i < module->descriptor->dependency_count; ++i) {
    const iree_vm_module_dependency_t* dependency =
        &module->descriptor->dependencies[i];
    IREE_RETURN_IF_ERROR(callback(user_data, dependency));
  }
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR iree_vm_native_module_get_import_function(
    iree_vm_native_module_t* module, iree_host_size_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  if (IREE_UNLIKELY(ordinal >= module->descriptor->import_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import ordinal out of range (0 < %zu < %zu)",
                            ordinal, module->descriptor->import_count);
  }
  const iree_vm_native_import_descriptor_t* import_descriptor =
      &module->descriptor->imports[ordinal];
  if (out_function) {
    out_function->module = &module->base_interface;
    out_function->linkage = iree_all_bits_set(import_descriptor->flags,
                                              IREE_VM_NATIVE_IMPORT_OPTIONAL)
                                ? IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL
                                : IREE_VM_FUNCTION_LINKAGE_IMPORT;
    out_function->ordinal = (uint16_t)ordinal;
  }
  if (out_name) {
    *out_name = import_descriptor->full_name;
  }
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR iree_vm_native_module_get_export_function(
    iree_vm_native_module_t* module, iree_host_size_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  if (IREE_UNLIKELY(ordinal >= module->descriptor->export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "export ordinal out of range (0 < %zu < %zu)",
                            ordinal, module->descriptor->export_count);
  }
  if (out_function) {
    out_function->module = &module->base_interface;
    out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
    out_function->ordinal = (uint16_t)ordinal;
  }
  const iree_vm_native_export_descriptor_t* export_descriptor =
      &module->descriptor->exports[ordinal];
  if (out_name) {
    *out_name = export_descriptor->local_name;
  }
  if (out_signature) {
    out_signature->calling_convention = export_descriptor->calling_convention;
  }
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR iree_vm_native_module_get_function(
    void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (out_function) memset(out_function, 0, sizeof(*out_function));
  if (out_name) memset(out_name, 0, sizeof(*out_name));
  if (out_signature) memset(out_signature, 0, sizeof(*out_signature));
  if (module->user_interface.get_function) {
    return module->user_interface.get_function(
        module->self, linkage, ordinal, out_function, out_name, out_signature);
  }
  switch (linkage) {
    case IREE_VM_FUNCTION_LINKAGE_IMPORT:
    case IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL:
      return iree_vm_native_module_get_import_function(
          module, ordinal, out_function, out_name, out_signature);
    case IREE_VM_FUNCTION_LINKAGE_EXPORT:
      return iree_vm_native_module_get_export_function(
          module, ordinal, out_function, out_name, out_signature);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "native modules do not support internal function queries");
  }
}

static iree_status_t IREE_API_PTR iree_vm_native_module_get_function_attr(
    void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
    iree_host_size_t index, iree_string_pair_t* out_attr) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.get_function_attr) {
    return module->user_interface.get_function_attr(module->self, linkage,
                                                    ordinal, index, out_attr);
  }
  // TODO(benvanik): implement native module reflection.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not yet implemented");
}

static iree_status_t IREE_API_PTR iree_vm_native_module_lookup_function(
    void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
    const iree_vm_function_signature_t* expected_signature,
    iree_vm_function_t* out_function) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  memset(out_function, 0, sizeof(*out_function));
  if (module->user_interface.lookup_function) {
    return module->user_interface.lookup_function(
        module->self, linkage, name, expected_signature, out_function);
  }

  if (IREE_UNLIKELY(linkage != IREE_VM_FUNCTION_LINKAGE_EXPORT)) {
    // NOTE: we could support imports if required.
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "native modules do not support import/internal function queries");
  }

  // Binary search through the export descriptors.
  ptrdiff_t min_ordinal = 0;
  ptrdiff_t max_ordinal = module->descriptor->export_count - 1;
  const iree_vm_native_export_descriptor_t* exports =
      module->descriptor->exports;
  while (min_ordinal <= max_ordinal) {
    ptrdiff_t ordinal = (min_ordinal + max_ordinal) / 2;
    int cmp = iree_string_view_compare(exports[ordinal].local_name, name);
    if (cmp == 0) {
      return iree_vm_native_module_get_function(self, linkage, ordinal,
                                                out_function, NULL, NULL);
    } else if (cmp < 0) {
      min_ordinal = ordinal + 1;
    } else {
      max_ordinal = ordinal - 1;
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND, "no function %.*s.%.*s exported by module",
      (int)module->descriptor->name.size, module->descriptor->name.data,
      (int)name.size, name.data);
}

static iree_status_t IREE_API_PTR
iree_vm_native_module_alloc_state(void* self, iree_allocator_t allocator,
                                  iree_vm_module_state_t** out_module_state) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  *out_module_state = NULL;
  if (module->user_interface.alloc_state) {
    return module->user_interface.alloc_state(module->self, allocator,
                                              out_module_state);
  }
  // Default to no state.
  return iree_ok_status();
}

static void IREE_API_PTR iree_vm_native_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.free_state) {
    module->user_interface.free_state(module->self, module_state);
    return;
  }
  // No-op in the default implementation.
  IREE_ASSERT_EQ(module_state, NULL);
}

static iree_status_t IREE_API_PTR iree_vm_native_module_resolve_import(
    void* self, iree_vm_module_state_t* module_state, iree_host_size_t ordinal,
    const iree_vm_function_t* function,
    const iree_vm_function_signature_t* signature) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.resolve_import) {
    return module->user_interface.resolve_import(module->self, module_state,
                                                 ordinal, function, signature);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "native module does not support imports");
}

static iree_status_t IREE_API_PTR iree_vm_native_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.notify) {
    return module->user_interface.notify(module->self, module_state, signal);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_native_module_issue_call(
    iree_vm_native_module_t* module, iree_vm_stack_t* stack,
    iree_vm_stack_frame_t* callee_frame, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage) {
  iree_vm_module_state_t* module_state = callee_frame->module_state;

  // Call the target function using the shim.
  const uint16_t function_ordinal = callee_frame->function.ordinal;
  const iree_vm_native_function_ptr_t* function_ptr =
      &module->descriptor->functions[function_ordinal];
  iree_status_t status =
      function_ptr->shim(stack, flags, args_storage, rets_storage,
                         function_ptr->target, module->self, module_state);
  if (iree_status_is_deferred(status)) {
    // Call deferred; bail and return to the scheduler.
    // Note that we preserve the stack.
    return status;
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS
    iree_string_view_t module_name IREE_ATTRIBUTE_UNUSED =
        iree_vm_native_module_name(module);
    iree_string_view_t function_name IREE_ATTRIBUTE_UNUSED =
        iree_string_view_empty();
    iree_status_ignore(iree_vm_native_module_get_export_function(
        module, function_ordinal, NULL, &function_name, NULL));
    return iree_status_annotate_f(status,
                                  "while invoking native function %.*s.%.*s",
                                  (int)module_name.size, module_name.data,
                                  (int)function_name.size, function_name.data);
#else
    return status;
#endif  // IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS
  }

  // Call completed successfully; pop the stack and return to caller.
  return iree_vm_stack_function_leave(stack);
}

static iree_status_t IREE_API_PTR iree_vm_native_module_begin_call(
    void* self, iree_vm_stack_t* stack, iree_vm_function_call_t call) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (IREE_UNLIKELY(call.function.linkage != IREE_VM_FUNCTION_LINKAGE_EXPORT) ||
      IREE_UNLIKELY(call.function.ordinal >=
                    module->descriptor->export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function ordinal out of bounds: 0 < %u < %zu",
                            call.function.ordinal,
                            module->descriptor->export_count);
  }
  if (module->user_interface.begin_call) {
    return module->user_interface.begin_call(module->self, stack, call);
  }

  // NOTE: VM stack is currently unused. We could stash things here for the
  // debugger or use it for coroutine state.
  iree_host_size_t frame_size = 0;

  iree_vm_stack_frame_t* callee_frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_stack_function_enter(
      stack, &call.function, IREE_VM_STACK_FRAME_NATIVE, frame_size,
      /*frame_cleanup_fn=*/NULL, &callee_frame));

  // Begin call with fresh callee frame.
  return iree_vm_native_module_issue_call(
      module, stack, callee_frame, IREE_VM_NATIVE_FUNCTION_CALL_BEGIN,
      call.arguments, call.results);  // tail
}

static iree_status_t IREE_API_PTR iree_vm_native_module_resume_call(
    void* self, iree_vm_stack_t* stack, iree_byte_span_t call_results) {
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)self;
  if (module->user_interface.resume_call) {
    return module->user_interface.resume_call(module->self, stack,
                                              call_results);
  }

  // Resume call using existing top frame.
  iree_vm_stack_frame_t* callee_frame = iree_vm_stack_top(stack);
  if (IREE_UNLIKELY(!callee_frame)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no frame at top of stack to resume");
  }
  return iree_vm_native_module_issue_call(
      module, stack, callee_frame, IREE_VM_NATIVE_FUNCTION_CALL_RESUME,
      iree_byte_span_empty(), call_results);  // tail
}

IREE_API_EXPORT iree_status_t iree_vm_native_module_create(
    const iree_vm_module_t* interface,
    const iree_vm_native_module_descriptor_t* module_descriptor,
    iree_vm_instance_t* instance, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(interface);
  IREE_ASSERT_ARGUMENT(module_descriptor);
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  if (IREE_UNLIKELY(!interface->begin_call) &&
      IREE_UNLIKELY(!module_descriptor->functions)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native modules must provide call support or function pointers");
  } else if (IREE_UNLIKELY(!interface->begin_call) &&
             IREE_UNLIKELY(module_descriptor->export_count !=
                           module_descriptor->function_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native modules using the default call support "
                            "must have 1:1 exports:function pointers");
  }

  // Perform some optional debug-only verification of the descriptor.
  // Since native modules are designed to be compiled in we don't need to do
  // this in release builds.
  IREE_RETURN_IF_ERROR(
      iree_vm_native_module_verify_descriptor(module_descriptor));

  // TODO(benvanik): invert allocation such that caller allocates and we init.
  // This would avoid the need for any dynamic memory allocation in the common
  // case as the outer user module interface could nest us. Note that we'd need
  // to expose this via a query_size function so that we could adjust the size
  // of our storage independent of the definition of the user module.
  iree_vm_native_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*module), (void**)&module));

  iree_status_t status =
      iree_vm_native_module_initialize(interface, module_descriptor, instance,
                                       allocator, (iree_vm_module_t*)module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, module);
    return status;
  }

  *out_module = &module->base_interface;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_native_module_initialize(
    const iree_vm_module_t* interface,
    const iree_vm_native_module_descriptor_t* module_descriptor,
    iree_vm_instance_t* instance, iree_allocator_t allocator,
    iree_vm_module_t* base_module) {
  IREE_ASSERT_ARGUMENT(interface);
  IREE_ASSERT_ARGUMENT(module_descriptor);
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(base_module);
  iree_vm_native_module_t* module = (iree_vm_native_module_t*)base_module;

  if (IREE_UNLIKELY(!interface->begin_call) &&
      IREE_UNLIKELY(!module_descriptor->functions)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native modules must provide call support or function pointers");
  } else if (IREE_UNLIKELY(!interface->begin_call) &&
             IREE_UNLIKELY(module_descriptor->export_count !=
                           module_descriptor->function_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native modules using the default call support "
                            "must have 1:1 exports:function pointers");
  }

  // Perform some optional debug-only verification of the descriptor.
  // Since native modules are designed to be compiled in we don't need to do
  // this in release builds.
  IREE_RETURN_IF_ERROR(
      iree_vm_native_module_verify_descriptor(module_descriptor));
  module->allocator = allocator;
  module->descriptor = module_descriptor;

  // TODO(benvanik): version interface and copy only valid bytes.
  memcpy(&module->user_interface, interface, sizeof(*interface));
  module->self =
      module->user_interface.self ? module->user_interface.self : module;

  // Base interface that routes through our thunks.
  iree_vm_module_initialize(&module->base_interface, module);
  module->base_interface.destroy = iree_vm_native_module_destroy;
  module->base_interface.name = iree_vm_native_module_name;
  module->base_interface.signature = iree_vm_native_module_signature;
  module->base_interface.get_module_attr =
      iree_vm_native_module_get_module_attr;
  module->base_interface.enumerate_dependencies =
      iree_vm_native_module_enumerate_dependencies;
  module->base_interface.lookup_function =
      iree_vm_native_module_lookup_function;
  module->base_interface.get_function = iree_vm_native_module_get_function;
  module->base_interface.get_function_attr =
      iree_vm_native_module_get_function_attr;
  module->base_interface.alloc_state = iree_vm_native_module_alloc_state;
  module->base_interface.free_state = iree_vm_native_module_free_state;
  module->base_interface.resolve_import = iree_vm_native_module_resolve_import;
  module->base_interface.notify = iree_vm_native_module_notify;
  module->base_interface.begin_call = iree_vm_native_module_begin_call;
  module->base_interface.resume_call = iree_vm_native_module_resume_call;

  return iree_ok_status();
}
