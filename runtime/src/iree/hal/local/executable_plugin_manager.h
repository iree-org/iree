// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_PLUGIN_MANAGER_H_
#define IREE_HAL_LOCAL_EXECUTABLE_PLUGIN_MANAGER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/executable_plugin.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_plugin_vtable_t
    iree_hal_executable_plugin_vtable_t;

typedef iree_status_t (*iree_hal_executable_plugin_resolve_thunk_t)(
    void* fn_ptr, void* self, const void* params, uint32_t* out_resolution);

// Interface for executable import plugins that provides a virtual destructor.
typedef struct iree_hal_executable_plugin_t {
  iree_atomic_ref_count_t ref_count;
  const iree_hal_executable_plugin_vtable_t* vtable;
  union {
    const iree_hal_executable_plugin_header_t** header;
    const iree_hal_executable_plugin_v0_t* v0;
  } library;
  void* self;
  iree_string_view_t identifier;
  iree_hal_executable_plugin_resolve_thunk_t resolve_thunk;
} iree_hal_executable_plugin_t;

// Initializes the base iree_hal_executable_plugin_t type using the given
// |header|. Called by subclasses upon allocating their storage.
//
// |required_features| are those the plugin must declare/use in order to load.
//
// If initialization fails it means the plugin could not be queried or was
// incompatible and the caller must clean up any resources prior to returning.
//
// |host_allocator| is passed to the plugin to service plugin allocations.
iree_status_t iree_hal_executable_plugin_initialize(
    const void* vtable, iree_hal_executable_plugin_features_t required_features,
    const iree_hal_executable_plugin_header_t** header_ptr,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_hal_executable_plugin_resolve_thunk_t resolve_thunk,
    iree_allocator_t host_allocator,
    iree_hal_executable_plugin_t* out_base_plugin);

// Retains the given |plugin| for the caller.
void iree_hal_executable_plugin_retain(iree_hal_executable_plugin_t* plugin);

// Releases the given |plugin| from the caller.
void iree_hal_executable_plugin_release(iree_hal_executable_plugin_t* plugin);

// Returns an import provider using the plugin to resolve imports.
// Note that the provider reference does not keep the plugin alive and the
// caller must ensure the plugin remains live for the duration of any provider
// references.
iree_hal_executable_import_provider_t iree_hal_executable_plugin_provider(
    iree_hal_executable_plugin_t* plugin);

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_manager_t
//===----------------------------------------------------------------------===//

// An import provider manager allowing for provider lifetime management and
// symbol resolution. The manager must be kept live for as long as any resolved
// symbol may be accessible.
//
// Thread-safe: registration is permanent and it is safe to register while loads
// are happening - though resolution behavior would be non-deterministic and
// it's strongly encouraged that all providers are registered upon startup.
typedef struct iree_hal_executable_plugin_manager_t
    iree_hal_executable_plugin_manager_t;

// Creates a new import manager with no registered providers.
// Storage for |capacity| registered providers will be allocated.
iree_status_t iree_hal_executable_plugin_manager_create(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_executable_plugin_manager_t** out_manager);

// Retains the given |manager| for the caller.
void iree_hal_executable_plugin_manager_retain(
    iree_hal_executable_plugin_manager_t* manager);

// Releases the given |manager| from the caller.
void iree_hal_executable_plugin_manager_release(
    iree_hal_executable_plugin_manager_t* manager);

// Registers the unowned |provider| at the top of the scan order.
// The caller must ensure the lifetime of the provider is longer than that of
// the manager.
iree_status_t iree_hal_executable_plugin_manager_register_provider(
    iree_hal_executable_plugin_manager_t* manager,
    iree_hal_executable_import_provider_t provider);

// Registers the |plugin| at the top of the scan order.
// The plugin will be retained for the lifetime of the manager.
iree_status_t iree_hal_executable_plugin_manager_register_plugin(
    iree_hal_executable_plugin_manager_t* manager,
    iree_hal_executable_plugin_t* plugin);

// Returns an import provider using the manager to resolve imports.
// Note that the provider reference does not keep the manager alive and the
// caller must ensure the manager remains live for the duration of any provider
// references.
iree_hal_executable_import_provider_t
iree_hal_executable_plugin_manager_provider(
    iree_hal_executable_plugin_manager_t* manager);

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_plugin_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_executable_plugin_t* plugin);
} iree_hal_executable_plugin_vtable_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_PLUGIN_MANAGER_H_
