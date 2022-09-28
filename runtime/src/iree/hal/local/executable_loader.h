// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_executable_import_provider_t
//===----------------------------------------------------------------------===//

// Interface used to resolve executable imports at load-time.
// This virtualizes some external provider and does not take ownership of the
// instance: callers must ensure that the provider remains valid for the
// lifetime of the executable loader that it is providing for.
typedef struct iree_hal_executable_import_provider_t {
  // TODO(benvanik): version field.
  IREE_API_UNSTABLE

  // User-defined pointer passed to all functions.
  void* self;

  // Resolves an import symbol with the given |symbol_name| and stores a pointer
  // to the function (or its context) in |out_fn_ptr|.
  iree_status_t(IREE_API_PTR* resolve)(void* self,
                                       iree_string_view_t symbol_name,
                                       void** out_fn_ptr,
                                       void** out_fn_context);
} iree_hal_executable_import_provider_t;

static inline iree_hal_executable_import_provider_t
iree_hal_executable_import_provider_null(void) {
  iree_hal_executable_import_provider_t provider = {NULL, NULL};
  return provider;
}

// Returns the import provider specified by
// IREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN or null.
//
// To use define a function like:
//   iree_hal_executable_import_provider_t my_provider(void) { ... }
// And define it:
//   -DIREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN=my_provider
iree_hal_executable_import_provider_t
iree_hal_executable_import_provider_default(void);

// Resolves an import symbol with the given |symbol_name| and stores a pointer
// to the function (or its context) in |out_fn_ptr|.
//
// A |symbol_name| ending in `?` indicates that the symbol is weak and is
// allowed to be resolved to NULL. Such cases will always return OK.
iree_status_t iree_hal_executable_import_provider_resolve(
    const iree_hal_executable_import_provider_t import_provider,
    iree_string_view_t symbol_name, void** out_fn_ptr, void** out_fn_context);

//===----------------------------------------------------------------------===//
// iree_hal_executable_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_loader_vtable_t
    iree_hal_executable_loader_vtable_t;

// Interface for compiled executable loader implementations.
// A loader may be as simple as something that resolves function pointers in the
// local executable for statically linked executables or as complex as a custom
// relocatable ELF loader. Loaders are registered and persist for each device
// they are attached to and may keep internal caches or memoize resources shared
// by multiple loaded executables.
//
// Thread-safe - multiple threads may load executables (including the *same*
// executable) simultaneously.
typedef struct iree_hal_executable_loader_t {
  iree_atomic_ref_count_t ref_count;
  const iree_hal_executable_loader_vtable_t* vtable;
  iree_hal_executable_import_provider_t import_provider;
} iree_hal_executable_loader_t;

// Initializes the base iree_hal_executable_loader_t type.
// Called by subclasses upon allocating their loader.
void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_import_provider_t import_provider,
    iree_hal_executable_loader_t* out_base_loader);

// Retains the given |executable_loader| for the caller.
void iree_hal_executable_loader_retain(
    iree_hal_executable_loader_t* executable_loader);

// Releases the given |executable_loader| from the caller.
void iree_hal_executable_loader_release(
    iree_hal_executable_loader_t* executable_loader);

// Returns true if the loader can load executables of the given
// |executable_format|. Note that loading may still fail if the executable uses
// features not available on the current host or runtime.
bool iree_hal_executable_loader_query_support(
    iree_hal_executable_loader_t* executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format);

// Returns true if any loader in the list can load executables of the given
// |executable_format|. Note that loading may still fail if the executable uses
// features not available on the current host or runtime.
bool iree_hal_query_any_executable_loader_support(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format);

// Tries loading the executable data provided in the given format.
// May fail even if the executable is valid if it requires features not
// supported by the current host or runtime (such as available architectures,
// imports, etc).
//
// Depending on loader ability the caching_mode is used to enable certain
// features such as instrumented profiling. Not all formats support these
// features and cooperation of both the compiler producing the executables and
// the runtime loader and system are required.
//
// Returns IREE_STATUS_CANCELLED when the loader cannot load the file in the
// given format.
iree_status_t iree_hal_executable_loader_try_load(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_loader_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_loader_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_executable_loader_t* executable_loader);

  bool(IREE_API_PTR* query_support)(
      iree_hal_executable_loader_t* executable_loader,
      iree_hal_executable_caching_mode_t caching_mode,
      iree_string_view_t executable_format);

  iree_status_t(IREE_API_PTR* try_load)(
      iree_hal_executable_loader_t* executable_loader,
      const iree_hal_executable_params_t* executable_params,
      iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable);
} iree_hal_executable_loader_vtable_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_
