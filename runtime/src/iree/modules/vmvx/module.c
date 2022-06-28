// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/vmvx/module.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/vm/api.h"

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_vmvx_module_t {
  iree_allocator_t host_allocator;
  // TODO(benvanik): types when we are not registering them globally.
} iree_vmvx_module_t;

#define IREE_VMVX_MODULE_CAST(module) \
  (iree_vmvx_module_t*)((uint8_t*)(module) + iree_vm_native_module_size());

typedef struct iree_vmvx_module_state_t {
  iree_allocator_t host_allocator;

  // If we have any external libraries we want to interact with that are
  // stateful we could store their state here. Note that VMVX invocations may
  // happen from any thread and concurrently and if the state is not thread-safe
  // we'll have to perform the synchronization ourselves here. That'd be bad,
  // of course, and an indication that whatever is being called is not suited
  // for this use.
} iree_vmvx_module_state_t;

static void IREE_API_PTR iree_vmvx_module_destroy(void* base_module) {
  // No state to clean up (yet).
}

static iree_status_t IREE_API_PTR
iree_vmvx_module_alloc_state(void* self, iree_allocator_t host_allocator,
                             iree_vm_module_state_t** out_module_state) {
  iree_vmvx_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR
iree_vmvx_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  iree_vmvx_module_state_t* state = (iree_vmvx_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

//===----------------------------------------------------------------------===//
// TODO
//===----------------------------------------------------------------------===//

// Placeholder to make the function pointer arrays happy (they can't be empty).
IREE_VM_ABI_EXPORT(iree_vmvx_module_placeholder,  //
                   iree_vmvx_module_state_t,      //
                   v, v) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_vmvx_module_exports_ table.
static const iree_vm_native_function_ptr_t iree_vmvx_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_vmvx_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_vmvx_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .reflection_attr_count = 0,                                  \
      .reflection_attrs = NULL,                                    \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_vmvx_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_vmvx_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_vmvx_module_descriptor_ = {
    .module_name = iree_string_view_literal("vmvx"),
    .import_count = 0,  // workaround for 0-length C struct
    .imports = iree_vmvx_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_vmvx_module_exports_),
    .exports = iree_vmvx_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_vmvx_module_funcs_),
    .functions = iree_vmvx_module_funcs_,
    .reflection_attr_count = 0,
    .reflection_attrs = NULL,
};

IREE_API_EXPORT iree_status_t iree_vmvx_module_create(
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_vmvx_module_destroy,
      .alloc_state = iree_vmvx_module_alloc_state,
      .free_state = iree_vmvx_module_free_state,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_vmvx_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_vmvx_module_descriptor_, host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_vmvx_module_t* module = IREE_VMVX_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}
