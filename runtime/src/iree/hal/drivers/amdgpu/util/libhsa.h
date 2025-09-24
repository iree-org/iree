// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_LIBHSA_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_LIBHSA_H_

#include "iree/base/api.h"

// HSA headers; note that these only somewhat assume shared library usage and
// though the functions are defined we cannot call them.
//
// TODO(benvanik): fork or upstream changes to the headers to make this safer?
// Similar to Vulkan we could have all the function declarations hidden and also
// include proper typed function pointers to aid in shared library use.
#include "third_party/hsa-runtime-headers/include/hsa/amd_hsa_queue.h"  // IWYU pragma: export
#include "third_party/hsa-runtime-headers/include/hsa/amd_hsa_signal.h"  // IWYU pragma: export
#include "third_party/hsa-runtime-headers/include/hsa/hsa.h"  // IWYU pragma: export
#include "third_party/hsa-runtime-headers/include/hsa/hsa_ext_amd.h"  // IWYU pragma: export
#include "third_party/hsa-runtime-headers/include/hsa/hsa_ven_amd_loader.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_dynamic_library_t iree_dynamic_library_t;
typedef struct iree_hal_amdgpu_libhsa_t iree_hal_amdgpu_libhsa_t;

//===----------------------------------------------------------------------===//
// Compile-time Configuration
//===----------------------------------------------------------------------===//

// By default we dynamically link against ROCR-Runtime. This allows us to keep
// our binary size small, version our runtime independently from the system
// software, and produce binaries that run on systems without ROCR-Runtime
// available.
//
// In the specific case of a user who is producing a binary that will only ever
// be run on a machine with an AMD GPU and up-to-date drivers we support
// statically linking against the installed ROCR-Runtime instead. This very
// slightly reduces overhead but is primarily useful for producing hermetic
// binaries.
//
// Set `-DIREE_HAL_AMDGPU_LIBHSA_STATIC=1` in CMake to link against the package.
// If already linking ROCR-Runtime into the top-level binary then add a compiler
// define `-DIREE_HAL_AMDGPU_LIBHSA_STATIC=1` to have this code emit imports
// (and then run LTO!).
#if !defined(IREE_HAL_AMDGPU_LIBHSA_STATIC)
#define IREE_HAL_AMDGPU_LIBHSA_STATIC 0
// For manual testing:
// #define IREE_HAL_AMDGPU_LIBHSA_STATIC 1
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC

#define IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_ALWAYS (1 << 0)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_SIGNALS (1 << 1)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_QUEUES (1 << 2)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_ALL  \
  (IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_ALWAYS |  \
   IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_SIGNALS | \
   IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_QUEUES)

// TODO(benvanik): expose as a more configurable setting - for now we try to
// stick to low-frequency events by default and have this for people debugging
// the implementation who want to see all calls.
#if !defined(IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE)
#if IREE_TRACING_FEATURES && !defined(NDEBUG)
#define IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE \
  IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_ALL
#else
#define IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE 0
#endif  // IREE_TRACING_FEATURES && !NDEBUG
#endif  // IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE

#if !defined(IREE_AMDGPU_DEVICE_PTR)
// Indicates a pointer is on the device. Used as annotations in host code.
// Device memory may not be host accessible or may have massive performance
// implications for accessing. This serves as a warning.
#define IREE_AMDGPU_DEVICE_PTR
#endif  // IREE_AMDGPU_DEVICE_PTR

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_libhsa_t
//===----------------------------------------------------------------------===//

typedef uint32_t iree_hal_amdgpu_libhsa_flags_t;
enum iree_hal_amdgpu_libhsa_flag_bits_e {
  IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE = 0u,
};

// Dynamically loaded libhsa-runtime64.so (or equivalent).
// Contains function pointers to resolved HSA API symbols as well as extension
// tables when available.
//
// Upon loading the `hsa_init` routine will be called. If there are other
// existing loaded HSA instances this will increase the reference count. Upon
// destruction `hsa_shut_down` will be called to decrement the HSA reference
// count. Within a single process HSA _should_ (and maybe _must_) be loaded from
// the same exact dynamic library; it's ok for different HAL drivers to load HSA
// redundantly so long as it's the same implementation.
//
// Because HSA ref counts itself we avoid doing so here. Each copy of the loaded
// library retains a reference and cloning it will always retain another. The
// structure can be treated by-value so long as the appropriate copy method is
// used when cloning. This design reduces one level of indirection into HSA: the
// library can be embedded in the device structure and function pointer offsets
// can be statically calculated. Does this matter in the grand scheme of things?
// Probably not.
//
// Thread-safe; immutable.
typedef struct iree_hal_amdgpu_libhsa_t {
  // True if hsa_init was called and hsa_shut_down must be called.
  bool initialized;

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC

  // Loaded HSA dynamic library.
  iree_dynamic_library_t* library;

#define IREE_HAL_AMDGPU_LIBHSA_PFN(trace_category, result_type, symbol, decl, \
                                   args)                                      \
  result_type(HSA_API* symbol)(decl);
#define DECL(...) __VA_ARGS__
#include "iree/hal/drivers/amdgpu/util/libhsa_tables.h"  // IWYU pragma: export

#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

  // HSA_EXTENSION_AMD_LOADER extension table.
  // Always required even when statically linking.
  hsa_ven_amd_loader_1_03_pfn_t amd_loader;
} iree_hal_amdgpu_libhsa_t;

// Initializes |out_libhsa| in-place with dynamically loaded HSA symbols.
// iree_hal_amdgpu_libhsa_deinitialize must be used to release the library
// resources. The populated structure is immutable once initialized but if
// copied the iree_hal_amdgpu_libhsa_copy API must be used (no memcpy!).
//
// |search_paths| will override the default library search paths and look for
// the canonical library file under each before falling back to the defaults.
// The `IREE_HAL_AMDGPU_LIBHSA_PATH` environment variable can also be set and
// will be checked after the explicitly provided search paths.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_libhsa_initialize(
    iree_hal_amdgpu_libhsa_flags_t flags, iree_string_view_list_t search_paths,
    iree_allocator_t host_allocator, iree_hal_amdgpu_libhsa_t* out_libhsa);

// Deinitializes |libhsa| by unloading the backing library. All function
// pointers will be invalidated, though be warned they may still be loaded and
// may appear to work (we can't reach out to clear cached pointers).
IREE_API_EXPORT void iree_hal_amdgpu_libhsa_deinitialize(
    iree_hal_amdgpu_libhsa_t* libhsa);

// Copies all resolved symbols from |libhsa| to |out_libhsa| and retains the
// HSA library. The target is assumed uninitialized.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_libhsa_copy(const iree_hal_amdgpu_libhsa_t* libhsa,
                            iree_hal_amdgpu_libhsa_t* out_libhsa);

// Appends the absolute path of the shared library or DLL providing the dynamic
// symbols.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_libhsa_append_path_to_builder(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_string_builder_t* builder);

// Returns an IREE status with the given |hsa_status| message formatted with
// textual information. |message| is optional.
IREE_API_EXPORT iree_status_t iree_status_from_hsa_status(
    const char* file, const uint32_t line, hsa_status_t hsa_status,
    const char* symbol, const char* message);

//===----------------------------------------------------------------------===//
// HSA API Wrappers
//===----------------------------------------------------------------------===//

// Wraps an iree_hal_amdgpu_libhsa_t* for use with the API wrappers.
// All calls should either embed their file and line information directly (if a
// thunk) or use this macro.
//
// Example:
//   iree_hal_amdgpu_libhsa_t* my_libhsa = ...;
//   iree_hsa_agent_get_info(IREE_LIBHSA(my_libhsa), agent, info, ...);
//   // ... expanded to ...
//   iree_hsa_agent_get_info(my_libhsa, __FILE__, __LINE__, agent, info, ...);
#define IREE_LIBHSA(libhsa) (libhsa), __FILE__, __LINE__

#define IREE_HAL_AMDGPU_LIBHSA_PFN_hsa_status_t(trace_category, result_type,  \
                                                symbol, decl, ...)            \
  iree_status_t iree_##symbol(                                                \
      const iree_hal_amdgpu_libhsa_t* IREE_RESTRICT libhsa, const char* file, \
      const uint32_t line _COMMA_DECL(decl));
#define IREE_HAL_AMDGPU_LIBHSA_PFN_result(trace_category, result_type, symbol, \
                                          decl, ...)                           \
  result_type iree_##symbol(                                                   \
      const iree_hal_amdgpu_libhsa_t* IREE_RESTRICT libhsa, const char* file,  \
      const uint32_t line _COMMA_DECL(decl));
#define IREE_HAL_AMDGPU_LIBHSA_PFN_void IREE_HAL_AMDGPU_LIBHSA_PFN_result
#define IREE_HAL_AMDGPU_LIBHSA_PFN_uint32_t IREE_HAL_AMDGPU_LIBHSA_PFN_result
#define IREE_HAL_AMDGPU_LIBHSA_PFN_uint64_t IREE_HAL_AMDGPU_LIBHSA_PFN_result
#define IREE_HAL_AMDGPU_LIBHSA_PFN_hsa_signal_value_t \
  IREE_HAL_AMDGPU_LIBHSA_PFN_result
#define IREE_HAL_AMDGPU_LIBHSA_PFN(trace_category, result_type, symbol, decl, \
                                   ...)                                       \
  IREE_HAL_AMDGPU_LIBHSA_PFN_##result_type(trace_category, result_type,       \
                                           symbol, DECL(decl), __VA_ARGS__)
#define DECL(...) __VA_ARGS__
#define _COMMA_DECL(...) __VA_OPT__(, ) __VA_ARGS__

#include "iree/hal/drivers/amdgpu/util/libhsa_tables.h"  // IWYU pragma: export

#undef _COMMA_DECL
#undef IREE_HAL_AMDGPU_LIBHSA_PFN_hsa_status_t
#undef IREE_HAL_AMDGPU_LIBHSA_PFN_result
#undef IREE_HAL_AMDGPU_LIBHSA_PFN_void
#undef IREE_HAL_AMDGPU_LIBHSA_PFN_uint32_t
#undef IREE_HAL_AMDGPU_LIBHSA_PFN_uint64_t
#undef IREE_HAL_AMDGPU_LIBHSA_PFN_hsa_signal_value_t

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_LIBHSA_H_
