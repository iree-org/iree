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

#ifndef IREE_HAL_EXECUTABLE_CACHE_H_
#define IREE_HAL_EXECUTABLE_CACHE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_layout.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_s iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Defines how the executable cache performs preparation.
enum iree_hal_executable_caching_mode_e {
  // Allows the cache to reference the provided executable_data after it has
  // prepared the executable. Callers must ensure the data remains valid for the
  // lifetime of the cache. If memory mapping constant executable data from
  // disk this can be used to avoid copies.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA = 1u << 0,
  // Allows the prepared executable to be cached persistently (on disk/etc).
  // Enable for any executable that is likely to be used in future runs.
  // Note that not all caches support persistent serialization and this is just
  // a hint.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING = 1u << 1,
  // Allows the cache to optimize the executable as much as it can.
  // This may cause preparation to take significantly longer while (hopefully)
  // improving runtime performance. Avoid for one-shot executables.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION = 1u << 2,
  // Enables Executable debugging methods if supported by the device and
  // executable. This may disable certain optimizations or retain additional
  // data to allow disassembly, stepping, etc.
  //
  // Device must support the IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING feature
  // and executables must support the ExecutableFeature::kDebugging feature.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_DEBUGGING = 1u << 3,
  // Enables Executable coverage if supported by the device and executable.
  // Depending on the optimization mode this may produce partial coverage
  // results (for example, when certain source operations were optimized away).
  //
  // Device must support the IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE feature
  // and executables must support the ExecutableFeature::kCoverage feature.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_COVERAGE = 1u << 4,
  // Enables Executable profiling if supported by the device and executable.
  // Depending on the optimization mode this may produce partial profiling
  // results. Profiling attribution (whether to the entire executable or
  // specific operations) depends on the implementation.
  //
  // Device must support the IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING feature
  // and executables must support the ExecutableFeature::kProfiling feature.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_PROFILING = 1u << 5,
  // Disables verification of executable layouts and modes.
  // This is useful when debugging with partial information but should never
  // be enabled for real usage as the verification is the best way to catch
  // API misuse.
  IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION = 1u << 6,
};
typedef uint32_t iree_hal_executable_caching_mode_t;

// Defines an executable compilation specification.
typedef struct {
  // Specifies what caching the executable cache is allowed to perform and
  // (if supported) which transformations on the executable contents are
  // allowed.
  iree_hal_executable_caching_mode_t caching_mode;

  // Indicates the format of the data in |executable_data|.
  iree_string_view_t executable_format;

  // Opaque compiler-generated executable data.
  // By default the memory storing the executable data is owned by the caller
  // and not guaranteed to live beyond the preparation call.
  //
  // Callers can indicate that they guarantee the lifetime of the memory
  // outlives the executable that will be created from it with the
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA flag, in which case
  // the cache is allowed to retain the data for as long as there is a reference
  // to any executable created using it still held by the caller.
  iree_const_byte_span_t executable_data;

  // A set of executable layouts for each entry point in the executable.
  // The order matches that produced by the compiler. As multiple entry points
  // may share the same layout some entries in this list may reference the same
  // executable layout objects.
  iree_host_size_t executable_layout_count;
  iree_hal_executable_layout_t* const* executable_layouts;
} iree_hal_executable_spec_t;

// Initializes |out_spec| to the default values for normal executables. Callers
// must override the fields as required.
void iree_hal_executable_spec_initialize(iree_hal_executable_spec_t* out_spec);

//===----------------------------------------------------------------------===//
// iree_hal_executable_cache_t
//===----------------------------------------------------------------------===//

// A cache of prepared executables for a particular device.
// Caches may be shared across multiple devices from the same driver or specific
// to individual devices. Caches may persist prepared executables across process
// launches or re-prepare them each run. Callers should assume that the cache is
// a no-op and the returned Executables only live for as long as the cache does.
//
// The term 'cache' here is rather optimistic - it's perfectly acceptable for
// implementations to not cache at all and return new Executables for each
// iree_hal_executable_cache_prepare_executable called (even for the same
// executable). Callers should expect such behavior and try to retain the
// results of the iree_hal_executable_cache_prepare_executable calls to reduce
// overhead in re-preparing executables.
//
// Thread-safe - multiple threads may prepare executables (including the *same*
// executable) simultaneously.
typedef struct iree_hal_executable_cache_s iree_hal_executable_cache_t;

// Creates an executable cache using the given identifier.
// The identifier is provided to the backing cache API as way to partition
// caches between different groups of executables (from different modules, etc).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_cache_create(
    iree_hal_device_t* device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache);

// Retains the given |executable_cache| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_executable_cache_retain(iree_hal_executable_cache_t* executable_cache);

// Releases the given |executable_cache| from the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_cache_release(
    iree_hal_executable_cache_t* executable_cache);

// Returns true if the executable cache can prepare the given executable input
// format. Preparation may still fail if the particular version or features
// required by the executable are not supported.
IREE_API_EXPORT bool IREE_API_CALL iree_hal_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format);

// Prepares the executable defined by |executable_spec| for use.
// The provided |executable_data| (in a format defined by |executable_format|)
// will be used to either lookup a previously prepared executable in the cache
// or prepare a new one.
//
// Each entry point in the executable requires a corresponding value in
// |executable_layouts| defining the layout used by the entry point. If multiple
// entry points use the same layouts they can reuse the same values.
//
// Depending on the driver preparation may take a non-trivial amount of time
// (such as when JITing/etc). As the cache is internally synchronized callers
// can issue preparation requests from multiple threads - even for the same
// executables - and calls will block until preparation completes.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* executable_cache,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_cache_t implementation details
//===----------------------------------------------------------------------===//

typedef struct {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_executable_cache_t* executable_cache);

  bool(IREE_API_PTR* can_prepare_format)(
      iree_hal_executable_cache_t* executable_cache,
      iree_hal_executable_caching_mode_t caching_mode,
      iree_string_view_t executable_format);

  iree_status_t(IREE_API_PTR* prepare_executable)(
      iree_hal_executable_cache_t* executable_cache,
      const iree_hal_executable_spec_t* executable_spec,
      iree_hal_executable_t** out_executable);
} iree_hal_executable_cache_vtable_t;

IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_cache_destroy(
    iree_hal_executable_cache_t* executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_CACHE_H_
