// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_PLUGIN_H_
#define IREE_HAL_LOCAL_EXECUTABLE_PLUGIN_H_

//===----------------------------------------------------------------------===//
//                                                                            //
//    ██╗░░░██╗███╗░░██╗░██████╗████████╗░█████╗░██████╗░██╗░░░░░███████╗     //
//    ██║░░░██║████╗░██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║░░░░░██╔════╝     //
//    ██║░░░██║██╔██╗██║╚█████╗░░░░██║░░░███████║██████╦╝██║░░░░░█████╗░░     //
//    ██║░░░██║██║╚████║░╚═══██╗░░░██║░░░██╔══██║██╔══██╗██║░░░░░██╔══╝░░     //
//    ╚██████╔╝██║░╚███║██████╔╝░░░██║░░░██║░░██║██████╦╝███████╗███████╗     //
//    ░╚═════╝░╚═╝░░╚══╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═════╝░╚══════╝╚══════╝     //
//                                                                            //
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_*_t
//===----------------------------------------------------------------------===//
// An interface providing lifetime management and import resolution for
// externally-defined executable library imports. Plugins can either be
// statically or dynamically linked into the runtime consuming them.
//
// Plugins only need to be used when lifetime management is required and
// otherwise users can register a much simpler import provider using
// iree_hal_executable_plugin_manager_register_provider. For plugins loaded
// from dynamic libraries, conditionally enabled, or needing lifetime management
// using this plugin API and iree_hal_executable_plugin_manager_register_plugin
// will ensure that import resolution is consistent and safe.
//
// Import resolution is intended to scale from function pointer lookup in .rdata
// to ahead-of-time JITs that may compile functions during resolution or return
// stubs that are JITed on-demand. See iree_hal_executable_plugin_v0_t::resolve
// for more information about the resolution process and how multiple plugins
// and import providers can resolve symbols.

// NOTE: this file is designed to be a standalone header: it is intended to be
// used by external projects that may build without the IREE source code.
// Include nothing but headers always available on all platforms/toolchains.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Runtime feature support metadata
//===----------------------------------------------------------------------===//

// Defines a bitfield of features that the plugin requires or supports.
enum iree_hal_executable_plugin_feature_bits_t {
  IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_NONE = 0u,

  // Plugin is built standalone and does not require any system facilities
  // beyond those provided by the plugin API.
  //
  // Nearly all plugins should be of this form in order to allow for portable
  // deployment and robust loading/execution; if a plugin is just a list of
  // pure functions there's very few ways it can fail at runtime.
  IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE = 1u << 0,

  // Plugin is built against the IREE runtime with IREE_STATUS_MODE > 0 and
  // iree_hal_executable_plugin_status_t will pass iree_status_t objects instead
  // of just status codes. The hosting runtime loading the plugin must also be
  // compiled with IREE_STATUS_MODE > 0.
  IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_FULL_STATUS = 1u << 1,
};
typedef uint32_t iree_hal_executable_plugin_features_t;

// Defines a set of supported sanitizers that plugins may be compiled with.
// The runtime uses this declaration to check as to whether the plugin is
// compatible with the hosting environment for cases where the sanitizer
// requires host support.
typedef enum iree_hal_executable_plugin_sanitizer_kind_e {
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE = 0,
  // Indicates the plugin is compiled to use AddressSanitizer:
  // https://clang.llvm.org/docs/AddressSanitizer.html
  // Equivalent compiler flag: -fsanitize=address
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_ADDRESS = 1,
  // Indicates the plugin is compiled to use MemorySanitizer:
  // https://clang.llvm.org/docs/MemorySanitizer.html
  // Equivalent compiler flag: -fsanitize=memory
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_MEMORY = 2,
  // Indicates the plugin is compiled to use ThreadSanitizer:
  // https://clang.llvm.org/docs/ThreadSanitizer.html
  // Equivalent compiler flag: -fsanitize=thread
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_THREAD = 3,
  // Indicates the plugin is compiled to use UndefinedBehaviorSanitizer:
  // https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
  // Equivalent compiler flag: -fsanitize=undefined
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_UNDEFINED = 4,

  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_MAX_ENUM = INT32_MAX,
} iree_hal_executable_plugin_sanitizer_kind_t;

// The sanitizer kind currently enabled in the compilation unit. Can be used
// in plugins to initialize the sanitizer header field.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND \
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_ADDRESS
#endif  // __has_feature(address_sanitizer)
#if __has_feature(memory_sanitizer)
#define IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND \
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_MEMORY
#endif  // __has_feature(memory_sanitizer)
#if __has_feature(thread_sanitizer)
#define IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND \
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_THREAD
#endif  // __has_feature(thread_sanitizer)
#endif  // defined(__has_feature)
#if !defined(IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND)
#define IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND \
  IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE
#endif  // !IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND

//===----------------------------------------------------------------------===//
// Versioning and interface querying
//===----------------------------------------------------------------------===//

// Version code indicating the minimum required runtime structures.
// Runtimes cannot load plugins with newer versions but may be able to load
// older versions if backward compatibility is enabled.
//
// NOTE: until we hit v1 the versioning scheme here is not set in stone.
// We may want to make this major release number, date codes (0x20220307),
// or some semantic versioning we track in whatever spec we end up having.
typedef uint32_t iree_hal_executable_plugin_version_t;

#define IREE_HAL_EXECUTABLE_PLUGIN_VERSION_0_1 0x00000001u

// The latest version of the plugin API; can be used to populate the
// iree_hal_executable_plugin_header_t::version when building plugins.
#define IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST \
  IREE_HAL_EXECUTABLE_PLUGIN_VERSION_0_1

// A header present at the top of all versions of the plugin API used by the
// runtime to ensure version compatibility.
typedef struct iree_hal_executable_plugin_header_t {
  // Version of the API this plugin was built with, which was likely the value
  // of IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST.
  iree_hal_executable_plugin_version_t version;

  // Name used for logging/diagnostics - should be C literal-like.
  const char* name;
  // Human-readable description used for logging/diagnostics.
  // Could contain configuration information (debug/release, features, etc),
  // implementation version, git commit sha of the build, etc.
  const char* description;

  // Bitfield of features required/supported by this plugin.
  iree_hal_executable_plugin_features_t features;

  // Which sanitizer the plugin is compiled to use, if any.
  // Plugins meant for use with a particular sanitizer will are only usable
  // with hosting code that is using the same sanitizer.
  iree_hal_executable_plugin_sanitizer_kind_t sanitizer;

  // Reserved for future use.
  uint64_t reserved[8];
} iree_hal_executable_plugin_header_t;

// Exported function from dynamic libraries for querying plugin information.
// This should be implemented as pure as possible and may be called many times
// while in process. This is only used to query for support, allow the plugin
// to perform version switching, and return the vtables the runtime needs to
// create and manage the plugin.
//
// The provided |max_version| is the maximum version the caller supports;
// callees must return NULL if their lowest available version is greater
// than the max version supported by the caller.
typedef const iree_hal_executable_plugin_header_t** (
    *iree_hal_executable_plugin_query_fn_t)(
    iree_hal_executable_plugin_version_t max_version, void* reserved);

// Function name exported from dynamic libraries (pass to dlsym).
#define IREE_HAL_EXECUTABLE_PLUGIN_EXPORT_NAME \
  "iree_hal_executable_plugin_query"

#if defined(_WIN32) || defined(__CYGWIN__)
#define IREE_HAL_EXECUTABLE_PLUGIN_EXPORT __declspec(dllexport)
#else
#define IREE_HAL_EXECUTABLE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_status_t
//===----------------------------------------------------------------------===//
//
// A lightweight shim of the minimal iree/base/status.h interface.
// This allows us to keep this header standalone for easy out-of-tree builds and
// version things separately in the future.
//
// Most simple plugins will likely be fine with numeric status codes but more
// complex ones may want the full status functionality in order to give back
// file/line, error messages, and annotations. If the plugin links against the
// IREE runtime it is allowed to return full iree_status_t objects across the
// boundary so long as the hosting runtime was compiled with IREE_STATUS_MODE >
// 0 (otherwise the runtime won't know how to deal with them). Plugins using
// this functionality must declare the feature bit
// IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_FULL_STATUS so that the runtime can
// verify it is compiled in a compatible mode.

// Well-known status codes matching iree_status_code_t.
// Note that any code within IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CODE_MASK is
// valid even if not enumerated here. Always check for unhandled errors/have
// default conditions.
typedef enum iree_hal_executable_plugin_status_code_e {
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_OK = 0,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CANCELLED = 1,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNKNOWN = 2,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_INVALID_ARGUMENT = 3,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_DEADLINE_EXCEEDED = 4,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND = 5,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_ALREADY_EXISTS = 6,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_PERMISSION_DENIED = 7,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_RESOURCE_EXHAUSTED = 8,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_FAILED_PRECONDITION = 9,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_ABORTED = 10,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_OUT_OF_RANGE = 11,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNIMPLEMENTED = 12,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_INTERNAL = 13,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNAVAILABLE = 14,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_DATA_LOSS = 15,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_UNAUTHENTICATED = 16,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_DEFERRED = 17,
  IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CODE_MASK = 0x1Fu,
} iree_hal_executable_plugin_status_code_t;

typedef struct iree_status_handle_t* iree_hal_executable_plugin_status_t;

#define iree_hal_executable_plugin_status_from_code(code)                                       \
  ((iree_hal_executable_plugin_status_t)((uintptr_t)((                                          \
                                             iree_hal_executable_plugin_status_code_t)(code)) & \
                                         IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CODE_MASK))

#define iree_hal_executable_plugin_status_code(value)           \
  ((iree_hal_executable_plugin_status_t)(((uintptr_t)(value)) & \
                                         IREE_HAL_EXECUTABLE_PLUGIN_STATUS_CODE_MASK))

#define iree_hal_executable_plugin_ok_status() \
  iree_hal_executable_plugin_status_from_code( \
      IREE_HAL_EXECUTABLE_PLUGIN_STATUS_OK)

#define iree_hal_executable_plugin_status_is_ok(value) \
  ((uintptr_t)(value) == IREE_HAL_EXECUTABLE_PLUGIN_STATUS_OK)

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_allocator_t
//===----------------------------------------------------------------------===//
//
// A lightweight shim of the iree/base/allocator.h interface.
// This allows us to keep this header standalone for easy out-of-tree builds and
// version things separately in the future.
//
// Plugins are expected to use the allocator for all requests such that the
// hosting runtime can track the allocations. Plugins built in systems that
// don't support custom allocators can do as they want but must not report
// IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE as the plugin would be making
// syscalls in order to allocate/free memory from the underlying system.
//
// The iree_hal_executable_plugin_allocator_t struct is compatible with
// iree_allocator_t and can be used interchangeably.

typedef enum iree_hal_executable_plugin_allocator_command_e {
  IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_MALLOC = 0,
  IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_CALLOC = 1,
  IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_REALLOC = 2,
  IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_FREE = 3,
} iree_hal_executable_plugin_allocator_command_t;

typedef struct iree_hal_executable_plugin_allocator_alloc_params_t {
  size_t byte_length;
} iree_hal_executable_plugin_allocator_alloc_params_t;

typedef iree_hal_executable_plugin_status_t (
    *iree_hal_executable_plugin_allocator_ctl_fn_t)(
    void* self, iree_hal_executable_plugin_allocator_command_t command,
    const void* params, void** inout_ptr);

typedef struct iree_hal_executable_plugin_allocator_t {
  void* self;
  iree_hal_executable_plugin_allocator_ctl_fn_t ctl;
} iree_hal_executable_plugin_allocator_t;

static iree_hal_executable_plugin_status_t
iree_hal_executable_plugin_allocator_malloc(
    iree_hal_executable_plugin_allocator_t allocator, size_t byte_length,
    void** inout_ptr) {
  iree_hal_executable_plugin_allocator_alloc_params_t params = {byte_length};
  return allocator.ctl(allocator.self,
                       IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_MALLOC,
                       &params, inout_ptr);
}

static iree_hal_executable_plugin_status_t
iree_hal_executable_plugin_allocator_free(
    iree_hal_executable_plugin_allocator_t allocator, void* ptr) {
  return allocator.ctl(allocator.self,
                       IREE_HAL_EXECUTABLE_PLUGIN_ALLOCATOR_COMMAND_FREE,
                       /*params=*/NULL, &ptr);
}

//===----------------------------------------------------------------------===//
// iree_hal_executable_plugin_string_view_t
//===----------------------------------------------------------------------===//

// iree_string_view_t-compatible type.
// The string data may not be NUL terminated and the provided size (in
// characters) must be used.
typedef struct iree_hal_executable_plugin_string_view_t {
  const char* data;
  size_t size;
} iree_hal_executable_plugin_string_view_t;

// iree_string_pair_t-compatible type.
typedef struct iree_hal_executable_plugin_string_pair_t {
  union {
    iree_hal_executable_plugin_string_view_t first;
    iree_hal_executable_plugin_string_view_t key;
  };
  union {
    iree_hal_executable_plugin_string_view_t second;
    iree_hal_executable_plugin_string_view_t value;
  };
} iree_hal_executable_plugin_string_pair_t;

//===----------------------------------------------------------------------===//
// Common utilities
//===----------------------------------------------------------------------===//

// Matches iree_hal_executable_import_resolution_bits_e.
// New bits may be added over time but the existing bits will not change.
enum iree_hal_executable_plugin_resolution_bits_e {
  // One or more missing optional symbols.
  IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL = 1u << 0,
};
typedef uint32_t iree_hal_executable_plugin_resolution_t;

// Returns true if the import |symbol_name| is optional.
static inline bool iree_hal_executable_plugin_import_is_optional(
    const char* symbol_name) {
  // A `?` prefix indicates the symbol is optional and can be NULL.
  // Since the strings are NUL terminated we know there's always 1 char and
  // we can just test that for the prefix.
  return symbol_name ? (symbol_name[0] == '?') : false;
}

static inline int iree_hal_executable_plugin_strcmp(const char* lhs,
                                                    const char* rhs) {
  unsigned char lhsc = 0;
  unsigned char rhsc = 0;
  do {
    lhsc = (unsigned char)*lhs++;
    rhsc = (unsigned char)*rhs++;
    if (lhsc == '\0') return lhsc - rhsc;
  } while (lhsc == rhsc);
  return lhsc - rhsc;
}

//===----------------------------------------------------------------------===//
// IREE_HAL_EXECUTABLE_PLUGIN_VERSION_0_*
//===----------------------------------------------------------------------===//

// Environment provided to the plugin on load.
typedef struct iree_hal_executable_plugin_environment_v0_t {
  // Allocator to be used for all plugin allocations for the lifetime of the
  // plugin (such as those needed during resolution). The allocator will be
  // valid until the plugin is unloaded.
  iree_hal_executable_plugin_allocator_t host_allocator;
} iree_hal_executable_plugin_environment_v0_t;

typedef struct iree_hal_executable_plugin_resolve_params_v0_t {
  size_t count;
  const char* const* symbol_names;
  void** out_fn_ptrs;
  void** out_fn_contexts;
} iree_hal_executable_plugin_resolve_params_v0_t;

typedef iree_hal_executable_plugin_status_t (
    *iree_hal_executable_plugin_resolve_fn_v0_t)(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution);

// Structure used for v0 plugin interfaces.
// The entire structure is designed to be read-only and able to live embedded in
// the binary .rdata section.
//
// Thread-safe: the plugin must be safe to load and resolve from multiple
// threads simultaneously. No unloads will be performed with active resolves.
typedef struct iree_hal_executable_plugin_v0_t {
  // Version/metadata header.
  // Will have a version of IREE_HAL_EXECUTABLE_PLUGIN_VERSION_*.
  const iree_hal_executable_plugin_header_t* header;

  // Loads a plugin by possibly allocating state, loading dependencies, and
  // preparing for resolution.
  //
  // The parameter strings are only available during the load and if the plugin
  // needs to retain any of them they must be cloned.
  //
  // The value specified in |out_self| will be passed to subsequent plugin
  // interface calls and may be NULL if the plugin has no state. As the same
  // plugin may be instantiated multiple times global state should be avoided
  // unless the
  iree_hal_executable_plugin_status_t (*load)(
      const iree_hal_executable_plugin_environment_v0_t* environment,
      size_t param_count,
      const iree_hal_executable_plugin_string_pair_t* params, void** out_self);

  // Unloads a plugin represented by |self|.
  // The plugin should free all resources as if in a shared library the hosting
  // runtime may immediately unload the library (dlclose/etc).
  void (*unload)(void* self);

  // Resolves |count| imports given |symbol_names| and stores pointers to their
  // implementation in |out_fn_ptrs| and optional contexts in |out_fn_contexts|.
  // The function contexts can be NULL, a pointer to shared state passed to each
  // import, unique state per function, or anything else. This allows JITs for
  // example to export a single function pointer and pack the information about
  // what to JIT in the context - or pass unique thunk functions for each import
  // and the shared JIT engine state for all contexts so that the thunk can
  // access it.
  //
  // A symbol name starting with `?` indicates that the symbol is optional and
  // is allowed to be resolved to NULL. Such cases will always return OK but set
  // the IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL resolution bit.
  //
  // Any already resolved function pointers will be skipped and left unmodified.
  // When there's only partial availability of required imports any available
  // ones will still be populated and NOT_FOUND will is returned. This allows
  // for looping over multiple providers to populate what they can and only
  // fails out if all providers return NOT_FOUND for a required import.
  //
  // The returned function pointers are the direct storage in the runtime.
  // Implementations are allowed to update their own entries (and _only_ their
  // entries) at runtime so long as they observe the thread-safety guarantees.
  // For example, a JIT may default all exports to JIT thunk functions and then
  // atomically swap them out for the translated function pointers as they are
  // available.
  //
  // Symbol names must be sorted alphabetically so if we cared we could use this
  // information to more efficiently resolve the symbols from providers (O(n)
  // walk vs potential O(nlogn)/O(n^2)). For example if JITing many similar
  // ukernel variants (matmul_Ma_Na_Ka, matmul_Mb_Nb_Kb, etc) a resolver can
  // avoid big switch tables.
  iree_hal_executable_plugin_resolve_fn_v0_t resolve;
} iree_hal_executable_plugin_v0_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_PLUGIN_H_
