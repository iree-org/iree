// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: native_module_test.h contains documented examples of how to use this!

#ifndef IREE_VM_NATIVE_MODULE_H_
#define IREE_VM_NATIVE_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes an imported native function in a native module.
// All of this information is assumed read-only and will be referenced for the
// lifetime of any module created with the descriptor.
typedef struct iree_vm_native_import_descriptor_t {
  // Fully-qualified function name (for example, 'other_module.foo').
  iree_string_view_t full_name;
} iree_vm_native_import_descriptor_t;

// Describes an exported native function in a native module.
// All of this information is assumed read-only and will be referenced for the
// lifetime of any module created with the descriptor.
typedef struct iree_vm_native_export_descriptor_t {
  // Module-local function name (for example, 'foo' for function 'module.foo').
  iree_string_view_t local_name;

  // Calling convention string; see iree/vm/module.h for details.
  iree_string_view_t calling_convention;

  // An optional list of function-level reflection attributes.
  iree_host_size_t reflection_attr_count;
  const iree_vm_reflection_attr_t* reflection_attrs;
} iree_vm_native_export_descriptor_t;

typedef iree_status_t(IREE_API_PTR* iree_vm_native_function_target_t)(
    iree_vm_stack_t* stack, void* module, void* module_state);

typedef iree_status_t(IREE_API_PTR* iree_vm_native_function_shim_t)(
    iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
    iree_vm_native_function_target_t target_fn, void* module,
    void* module_state, iree_vm_execution_result_t* out_result);

// An entry in the function pointer table.
typedef struct iree_vm_native_function_ptr_t {
  // A shim function that takes the VM ABI and maps it to the target ABI.
  iree_vm_native_function_shim_t shim;
  // Target function passed to the shim.
  iree_vm_native_function_target_t target;
} iree_vm_native_function_ptr_t;

// Describes a native module implementation by way of descriptor tables.
// All of this information is assumed read-only and will be referenced for the
// lifetime of any module created with the descriptor.
//
// The common native module code will use this descriptor to return metadata on
// query, lookup exported functions, and call module-provided implementation
// functions for state and call management.
typedef struct iree_vm_native_module_descriptor_t {
  IREE_API_UNSTABLE

  // Name of the module prefixed on all exported functions.
  iree_string_view_t module_name;

  // All imported function descriptors.
  // interface.resolve_import will be called for each import.
  // Imports must be in order sorted by name compatible with
  // iree_string_view_compare.
  iree_host_size_t import_count;
  const iree_vm_native_import_descriptor_t* imports;

  // All exported function descriptors.
  // Exports must be in order sorted by name compatible with
  // iree_string_view_compare.
  iree_host_size_t export_count;
  const iree_vm_native_export_descriptor_t* exports;

  // All function shims and target function pointers.
  // These must match 1:1 with the exports if using the default begin_call
  // implementation and are optional if overriding begin_call.
  iree_host_size_t function_count;
  const iree_vm_native_function_ptr_t* functions;

  // An optional list of module-level reflection attributes.
  iree_host_size_t reflection_attr_count;
  const iree_vm_reflection_attr_t* reflection_attrs;
} iree_vm_native_module_descriptor_t;

// Returns the size, in bytes, of the allocation required for native modules.
// Callers may allocate more memory if they need additional storage.
IREE_API_EXPORT iree_host_size_t iree_vm_native_module_size(void);

// Creates a new native module with the metadata tables in |descriptor|.
// These tables will be used for reflection and function lookup, and the
// provided function pointers will be called when state needs to be managed or
// exported functions need to be called.
//
// An implementation |interface| providing functions for state management and
// function calls can be provided to override default implementations of
// functions. The structure will be copied and the self pointer will be passed
// to all |interface| functions.
//
// The provided |descriptor| will be referenced by the created module and must
// be kept live for the lifetime of the module.
IREE_API_EXPORT iree_status_t iree_vm_native_module_create(
    const iree_vm_module_t* interface,
    const iree_vm_native_module_descriptor_t* module_descriptor,
    iree_allocator_t allocator, iree_vm_module_t** out_module);

IREE_API_EXPORT iree_status_t iree_vm_native_module_initialize(
    const iree_vm_module_t* interface,
    const iree_vm_native_module_descriptor_t* module_descriptor,
    iree_allocator_t allocator, iree_vm_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_NATIVE_MODULE_H_
