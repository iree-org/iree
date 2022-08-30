// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_EXECUTABLE_CACHE_H_
#define IREE_HAL_EXECUTABLE_CACHE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/executable.h"
#include "iree/hal/pipeline_layout.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Defines how the executable cache performs preparation.
enum iree_hal_executable_caching_mode_bits_t {
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
  // Disables verification of pipeline layouts and modes.
  // This is useful when debugging with partial information but should never
  // be enabled for real usage as the verification is the best way to catch
  // API misuse.
  IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION = 1u << 6,
};
typedef uint32_t iree_hal_executable_caching_mode_t;

// Defines an executable compilation specification.
typedef struct iree_hal_executable_params_t {
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

  // A set of pipeline layouts for each entry point in the executable.
  // The order matches that produced by the compiler. As multiple entry points
  // may share the same layout some entries in this list may reference the same
  // pipeline layout objects.
  iree_host_size_t pipeline_layout_count;
  iree_hal_pipeline_layout_t* const* pipeline_layouts;

  // Executable-level constants table used to perform runtime specialization
  // when information is not available statically during compilation. The
  // compiler defines the contents of the table, how they are populated, and
  // their usage in the executable.
  //
  // For targets that natively support specialization these directly map down:
  //   Metal: function constants
  //   WGSL: pipeline overrides
  //   Vulkan/SPIR-V: specialization constants
  // Other targets may present these as constant tables or uniform buffers.
  // Since the values cannot change after initialization targets that JIT may
  // perform substitution during initialization to inline the values
  // immediately (via CUDA PTX linking, etc).
  iree_host_size_t constant_count;
  const uint32_t* constants;
} iree_hal_executable_params_t;

// Initializes |out_executable_params| to the default values for normal
// executables. Callers must override the fields as required.
void iree_hal_executable_params_initialize(
    iree_hal_executable_params_t* out_executable_params);

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
typedef struct iree_hal_executable_cache_t iree_hal_executable_cache_t;

// Creates an executable cache using the given identifier.
// The identifier is provided to the backing cache API as way to partition
// caches between different groups of executables (from different modules, etc).
//
// Any host-side work that needs to be performed will be scheduled on |loop|.
// This enables JITs, device-specific translation, and verification to be
// parallelized using a shared scheduler. The loop must remain valid for the
// lifetime of the executable cache.
IREE_API_EXPORT iree_status_t iree_hal_executable_cache_create(
    iree_hal_device_t* device, iree_string_view_t identifier, iree_loop_t loop,
    iree_hal_executable_cache_t** out_executable_cache);

// Retains the given |executable_cache| for the caller.
IREE_API_EXPORT void iree_hal_executable_cache_retain(
    iree_hal_executable_cache_t* executable_cache);

// Releases the given |executable_cache| from the caller.
IREE_API_EXPORT void iree_hal_executable_cache_release(
    iree_hal_executable_cache_t* executable_cache);

// Returns true if the executable cache can prepare the given executable input
// format. Preparation may still fail if the particular version or features
// required by the executable are not supported.
IREE_API_EXPORT bool iree_hal_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format);

// Prepares the executable defined by |executable_params| for use.
// The provided |executable_data| (in a format defined by |executable_format|)
// will be used to either lookup a previously prepared executable in the cache
// or prepare a new one.
//
// Each entry point in the executable requires a corresponding value in
// |pipeline_layouts| defining the layout used by the entry point. If multiple
// entry points use the same layouts they can reuse the same values.
//
// Depending on the driver preparation may take a non-trivial amount of time
// (such as when JITing/etc). As the cache is internally synchronized callers
// can issue preparation requests from multiple threads - even for the same
// executables - and calls will block until preparation completes.
IREE_API_EXPORT iree_status_t iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_cache_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_cache_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_executable_cache_t* executable_cache);

  bool(IREE_API_PTR* can_prepare_format)(
      iree_hal_executable_cache_t* executable_cache,
      iree_hal_executable_caching_mode_t caching_mode,
      iree_string_view_t executable_format);

  iree_status_t(IREE_API_PTR* prepare_executable)(
      iree_hal_executable_cache_t* executable_cache,
      const iree_hal_executable_params_t* executable_params,
      iree_hal_executable_t** out_executable);
} iree_hal_executable_cache_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_executable_cache_vtable_t);

IREE_API_EXPORT void iree_hal_executable_cache_destroy(
    iree_hal_executable_cache_t* executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_CACHE_H_
