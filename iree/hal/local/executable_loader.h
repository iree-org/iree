// Copyright 2020 Google LLC
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
// iree_hal_executable_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_loader_vtable_s
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
typedef struct {
  iree_atomic_ref_count_t ref_count;
  const iree_hal_executable_loader_vtable_t* vtable;
} iree_hal_executable_loader_t;

// Initializes the base iree_hal_executable_loader_t type.
// Called by subclasses upon allocating their loader.
void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_loader_t* out_base_loader);

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
    iree_hal_executable_format_t executable_format);

// Tries loading the |executable_data| provided in the given
// |executable_format|. May fail even if the executable is valid if it requires
// features not supported by the current host or runtime (such as available
// architectures, imports, etc).
//
// Depending on loader ability the |caching_mode| is used to enable certain
// features such as instrumented profiling. Not all formats support these
// features and cooperation of both the compiler producing the executables and
// the runtime loader and system are required.
//
// Returns IREE_STATUS_CANCELLED when the loader cannot load the file in the
// given format.
iree_status_t iree_hal_executable_loader_try_load(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_loader_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_loader_vtable_s {
  void(IREE_API_PTR* destroy)(iree_hal_executable_loader_t* executable_loader);

  bool(IREE_API_PTR* query_support)(
      iree_hal_executable_loader_t* executable_loader,
      iree_hal_executable_caching_mode_t caching_mode,
      iree_hal_executable_format_t executable_format);

  iree_status_t(IREE_API_PTR* try_load)(
      iree_hal_executable_loader_t* executable_loader,
      const iree_hal_executable_spec_t* executable_spec,
      iree_hal_executable_t** out_executable);
} iree_hal_executable_loader_vtable_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_
