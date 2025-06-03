// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"

//===----------------------------------------------------------------------===//
// hsa_status_t interop
//===----------------------------------------------------------------------===//

// Maps an HSA status to an IREE status as best we can.
static iree_status_code_t iree_hsa_status_code(hsa_status_t status) {
  switch (status) {
    default:
      return IREE_STATUS_UNKNOWN;
    case HSA_STATUS_SUCCESS:
      return IREE_STATUS_OK;
    case HSA_STATUS_INFO_BREAK:
      return IREE_STATUS_CANCELLED;
    case HSA_STATUS_ERROR:
      return IREE_STATUS_UNKNOWN;
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
    case HSA_STATUS_ERROR_INVALID_AGENT:
    case HSA_STATUS_ERROR_INVALID_REGION:
    case HSA_STATUS_ERROR_INVALID_SIGNAL:
    case HSA_STATUS_ERROR_INVALID_QUEUE:
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
    case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
    case HSA_STATUS_ERROR_INVALID_FILE:
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
    case HSA_STATUS_ERROR_INVALID_CACHE:
    case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
    case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
    case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return IREE_STATUS_INVALID_ARGUMENT;
    case HSA_STATUS_ERROR_RESOURCE_FREE:
      return IREE_STATUS_INTERNAL;
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
    case HSA_STATUS_ERROR_INVALID_ISA:
      return IREE_STATUS_INCOMPATIBLE;
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case HSA_STATUS_ERROR_NOT_INITIALIZED:
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return IREE_STATUS_FAILED_PRECONDITION;
    case HSA_STATUS_ERROR_INVALID_INDEX:
      return IREE_STATUS_OUT_OF_RANGE;
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return IREE_STATUS_NOT_FOUND;
    case HSA_STATUS_ERROR_EXCEPTION:
    case HSA_STATUS_ERROR_FATAL:
      return IREE_STATUS_DATA_LOSS;
  }
}

// Returns the stringified form of the HSA status enum value.
static const char* iree_hsa_status_string(hsa_status_t status) {
  switch (status) {
    default:
      return "?";
    case HSA_STATUS_SUCCESS:
      return "HSA_STATUS_SUCCESS";
    case HSA_STATUS_INFO_BREAK:
      return "HSA_STATUS_INFO_BREAK";
    case HSA_STATUS_ERROR:
      return "HSA_STATUS_ERROR";
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
      return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
      return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
      return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case HSA_STATUS_ERROR_INVALID_AGENT:
      return "HSA_STATUS_ERROR_INVALID_AGENT";
    case HSA_STATUS_ERROR_INVALID_REGION:
      return "HSA_STATUS_ERROR_INVALID_REGION";
    case HSA_STATUS_ERROR_INVALID_SIGNAL:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case HSA_STATUS_ERROR_INVALID_QUEUE:
      return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
      return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case HSA_STATUS_ERROR_RESOURCE_FREE:
      return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case HSA_STATUS_ERROR_NOT_INITIALIZED:
      return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case HSA_STATUS_ERROR_INVALID_INDEX:
      return "HSA_STATUS_ERROR_INVALID_INDEX";
    case HSA_STATUS_ERROR_INVALID_ISA:
      return "HSA_STATUS_ERROR_INVALID_ISA";
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:
      return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
      return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    case HSA_STATUS_ERROR_EXCEPTION:
      return "HSA_STATUS_ERROR_EXCEPTION";
    case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL";
    case HSA_STATUS_ERROR_INVALID_FILE:
      return "HSA_STATUS_ERROR_INVALID_FILE";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER";
    case HSA_STATUS_ERROR_INVALID_CACHE:
      return "HSA_STATUS_ERROR_INVALID_CACHE";
    case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
      return "HSA_STATUS_ERROR_INVALID_WAVEFRONT";
    case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP";
    case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE";
    case HSA_STATUS_ERROR_FATAL:
      return "HSA_STATUS_ERROR_FATAL";
  }
}

// Returns a string literal describing the given HSA status code.
// We inline the string mapping here instead of calling hsa_status_string so
// that we can customize the error messages to our uses (and avoid the need for
// carrying the libhsa pointer to resolve statuses).
static const char* iree_hsa_status_description(hsa_status_t status) {
  switch (status) {
    default:
      return "Unknown.";
    case HSA_STATUS_SUCCESS:
      return "The function has been executed successfully.";
    case HSA_STATUS_INFO_BREAK:
      return "A traversal over a list of elements has been interrupted by the "
             "application before completing.";
    case HSA_STATUS_ERROR:
      return "A generic error has occurred.";
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
      return "One of the actual arguments does not meet a precondition stated "
             "in the documentation of the corresponding formal argument.";
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
      return "The requested queue creation is not valid.";
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
      return "The requested allocation is not valid.";
    case HSA_STATUS_ERROR_INVALID_AGENT:
      return "The agent is invalid.";
    case HSA_STATUS_ERROR_INVALID_REGION:
      return "The memory region is invalid.";
    case HSA_STATUS_ERROR_INVALID_SIGNAL:
      return "The signal is invalid.";
    case HSA_STATUS_ERROR_INVALID_QUEUE:
      return "The queue is invalid.";
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
      return "The HSA runtime failed to allocate the necessary resources. This "
             "error may also occur when the HSA runtime needs to spawn threads "
             "or create internal OS-specific events.";
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      return "The AQL packet is malformed.";
    case HSA_STATUS_ERROR_RESOURCE_FREE:
      return "An error has been detected while releasing a resource.";
    case HSA_STATUS_ERROR_NOT_INITIALIZED:
      return "An API other than hsa_init has been invoked while the reference "
             "count of the HSA runtime is 0.";
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return "The maximum reference count for the object has been reached.";
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      return "The arguments passed to a functions are not compatible.";
    case HSA_STATUS_ERROR_INVALID_INDEX:
      return "The index is invalid.";
    case HSA_STATUS_ERROR_INVALID_ISA:
      return "The instruction set architecture is invalid.";
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:
      return "The instruction set architecture name is invalid.";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      return "The code object is invalid.";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
      return "The executable is invalid.";
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
      return "The executable is frozen.";
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return "There is no symbol with the given name.";
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
      return "The variable is already defined.";
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return "The variable is undefined.";
    case HSA_STATUS_ERROR_EXCEPTION:
      return "An HSAIL operation resulted in a hardware exception.";
    case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
      return "The code object symbol is invalid.";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
      return "The executable symbol is invalid.";
    case HSA_STATUS_ERROR_INVALID_FILE:
      return "The file descriptor is invalid.";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
      return "The code object reader is invalid.";
    case HSA_STATUS_ERROR_INVALID_CACHE:
      return "The cache is invalid.";
    case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
      return "The wavefront is invalid.";
    case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
      return "The signal group is invalid.";
    case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return "The HSA runtime is not in the configuration state.";
    case HSA_STATUS_ERROR_FATAL:
      return "The queue received an error that may require process "
             "termination.";
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_libhsa_t
//===----------------------------------------------------------------------===//

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC

static iree_status_t iree_hal_amdgpu_libhsa_load_symbols(
    iree_dynamic_library_t* library, iree_hal_amdgpu_libhsa_t* out_libhsa) {
#define IREE_HAL_AMDGPU_LIBHSA_PFN(trace_category, result_type, symbol, ...) \
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(                   \
      library, #symbol, (void**)&out_libhsa->symbol));
#include "iree/hal/drivers/amdgpu/util/libhsa_tables.h"  // IWYU pragma: keep
  return iree_ok_status();
}

// TODO(benvanik): move someplace central - status and string_builder have a
// cycle and it's tricky to move there right now.
static bool iree_string_builder_append_status(iree_string_builder_t* builder,
                                              iree_status_t status) {
  // Calculate total length minus the NUL terminator.
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return false;
  }

  // Buffer capacity needs to be +1 for the NUL terminator (see snprintf).
  char* buffer = NULL;
  iree_status_t append_status =
      iree_string_builder_append_inline(builder, buffer_length, &buffer);
  if (!iree_status_is_ok(append_status)) {
    iree_status_ignore(append_status);
    return false;
  }
  if (!buffer) {
    // Size calculation mode; builder has been updated but no need to format.
    return true;
  }

  // Format into the buffer.
  return iree_status_format(status, buffer_length + 1, buffer, &buffer_length);
}

static bool iree_hal_amdgpu_libhsa_try_load_library_from_file(
    iree_hal_amdgpu_libhsa_flags_t flags, const char* file_path,
    iree_string_builder_t* error_builder, iree_allocator_t host_allocator,
    iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, file_path);
  *out_library = NULL;

  // Try loading from the given file path.
  iree_status_t status = iree_dynamic_library_load_from_file(
      file_path, flags, host_allocator, out_library);

  // Append error message to the status builder.
  if (!iree_status_is_ok(status)) {
    IREE_IGNORE_ERROR(iree_string_builder_append_format(
        error_builder, "\n  Tried: %s\n    ", file_path));
    iree_string_builder_append_status(error_builder, status);
  }

  iree_status_ignore(status);
  IREE_TRACE_ZONE_END(z0);
  return *out_library != NULL;
}

static const char* iree_hal_amdgpu_libhsa_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    // NOTE: this doesn't exist (yet) and will never resolve. Whenever HSA
    // appears on Windows again (🤞) we'll update this to point at it. For now
    // users can still build the HAL driver but it won't run.
    "hsa-runtime64.dll",
#else
    "libhsa-runtime64.so",
#endif  // IREE_PLATFORM_WINDOWS
};

static bool iree_hal_amdgpu_libhsa_try_load_library_from_path(
    iree_hal_amdgpu_libhsa_flags_t flags, iree_string_view_t path_fragment,
    iree_string_builder_t* error_builder, iree_allocator_t host_allocator,
    iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path_fragment.data, path_fragment.size);
  *out_library = NULL;

  // System APIs need cstrings and we need to perform some path manipulation so
  // we do that locally in a heap-allocated NUL-terminated string builder.
  iree_string_builder_t path_builder;
  iree_string_builder_initialize(host_allocator, &path_builder);

  if (iree_file_path_is_dynamic_library(path_fragment)) {
    // User provided a filename - try to use it directly. If it's an absolute
    // file path the system will try that and otherwise it'll search all library
    // paths for the given filename.
    iree_status_ignore(
        iree_string_builder_append_string(&path_builder, path_fragment));
    iree_hal_amdgpu_libhsa_try_load_library_from_file(
        flags, iree_string_builder_buffer(&path_builder), error_builder,
        host_allocator, out_library);
  } else {
    // Join the provided path with each canonical name and try that.
    iree_string_builder_reset(&path_builder);
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_hal_amdgpu_libhsa_names) && !*out_library;
         ++i) {
      iree_status_ignore(iree_string_builder_append_format(
          &path_builder, "%.*s/%s", (int)path_fragment.size, path_fragment.data,
          iree_hal_amdgpu_libhsa_names[i]));
      path_builder.size = iree_file_path_canonicalize(
          (char*)iree_string_builder_buffer(&path_builder),
          iree_string_builder_size(&path_builder));
      iree_hal_amdgpu_libhsa_try_load_library_from_file(
          flags, iree_string_builder_buffer(&path_builder), error_builder,
          host_allocator, out_library);
    }
  }

  iree_string_builder_deinitialize(&path_builder);

  IREE_TRACE_ZONE_END(z0);
  return *out_library != NULL;
}

static iree_status_t iree_hal_amdgpu_libhsa_load_library(
    iree_hal_amdgpu_libhsa_flags_t flags, iree_string_view_list_t search_paths,
    iree_allocator_t host_allocator, iree_hal_amdgpu_libhsa_t* out_libhsa) {
  IREE_ASSERT_ARGUMENT(out_libhsa);
  IREE_TRACE_ZONE_BEGIN(z0);
  out_libhsa->library = NULL;

  // Accumulate error messages for each path and library tried. Psychic
  // debugging user library paths is painful and this should give enough info
  // for them to figure it out themselves.
  iree_string_builder_t error_builder;
  iree_string_builder_initialize(host_allocator, &error_builder);

  iree_dynamic_library_t* library = NULL;

  // If the caller provided explicit paths we always try to use those first.
  // This allows a hosting application to handle overrides as they see fit.
  for (iree_host_size_t i = 0; i < search_paths.count && !library; ++i) {
    iree_hal_amdgpu_libhsa_try_load_library_from_path(
        flags, search_paths.values[i], &error_builder, host_allocator,
        &library);
  }

  // If no user path provided the library try the environment variable.
  // This allows for users to inject a specific path without needing to
  // recompile their application/pass through flags down into IREE.
  iree_string_view_t env_path =
      iree_make_cstring_view(getenv("IREE_HAL_AMDGPU_LIBHSA_PATH"));
  if (!library && !iree_string_view_is_empty(env_path)) {
    iree_hal_amdgpu_libhsa_try_load_library_from_path(
        flags, env_path, &error_builder, host_allocator, &library);
  }

  // Fallback (that is the common case) and try loading with the canonical
  // library names from the system search paths.
  if (!library) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_hal_amdgpu_libhsa_names) && !library; ++i) {
      if (iree_hal_amdgpu_libhsa_try_load_library_from_file(
              flags, iree_hal_amdgpu_libhsa_names[i], &error_builder,
              host_allocator, &library)) {
        break;
      }
    }
  }

  // If no library was found emit the full failure status.
  iree_status_t status = iree_ok_status();
  if (!library) {
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "HSA/ROCR-Runtime library not found; ensure it is "
                         "installed and on a valid search path (or specified "
                         "with IREE_HAL_AMDGPU_LIBHSA_PATH): %.*s",
                         (int)iree_string_builder_size(&error_builder),
                         iree_string_builder_buffer(&error_builder));
  }
  iree_string_builder_deinitialize(&error_builder);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_libhsa_load_symbols(library, out_libhsa);
    if (!iree_status_is_ok(status) && out_libhsa->hsa_init) {
      iree_string_builder_t annotation_builder;
      iree_string_builder_initialize(host_allocator, &annotation_builder);
      IREE_IGNORE_ERROR(iree_dynamic_library_append_symbol_path_to_builder(
          out_libhsa->hsa_init, &annotation_builder));
      status = iree_status_annotate_f(
          status, "using %.*s",
          (int)iree_string_builder_size(&annotation_builder),
          iree_string_builder_buffer(&annotation_builder));
      iree_string_builder_deinitialize(&annotation_builder);
    }
  }

  if (iree_status_is_ok(status)) {
    out_libhsa->library = library;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_libhsa_unload_library(
    iree_hal_amdgpu_libhsa_t* libhsa) {
  iree_dynamic_library_release(libhsa->library);
}

static void iree_hal_amdgpu_libhsa_copy_library(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_libhsa_t* out_libhsa) {
  // Copy all pointers.
  memcpy(out_libhsa, libhsa, sizeof(*out_libhsa));

  // Retain the library. The target libhsa_t will release it on unload.
  iree_dynamic_library_retain(out_libhsa->library);
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_libhsa_append_path_to_builder(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_string_builder_t* builder) {
  // Using a symbol we know is always present.
  return iree_dynamic_library_append_symbol_path_to_builder(libhsa->hsa_init,
                                                            builder);
}

#else
static iree_status_t iree_hal_amdgpu_libhsa_load_library(
    iree_hal_amdgpu_libhsa_flags_t flags, iree_string_view_list_t search_paths,
    iree_allocator_t host_allocator, iree_hal_amdgpu_libhsa_t* out_libhsa) {
  return iree_ok_status();
}
static void iree_hal_amdgpu_libhsa_unload_library(
    iree_hal_amdgpu_libhsa_t* libhsa) {}
static void iree_hal_amdgpu_libhsa_copy_library(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_libhsa_t* out_libhsa) {
  memcpy(out_libhsa, libhsa, sizeof(*out_libhsa));
}
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_libhsa_append_path_to_builder(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "<statically linked>");
}
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_libhsa_initialize(
    iree_hal_amdgpu_libhsa_flags_t flags, iree_string_view_list_t search_paths,
    iree_allocator_t host_allocator, iree_hal_amdgpu_libhsa_t* out_libhsa) {
  IREE_ASSERT_ARGUMENT(out_libhsa);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, flags);

  // Load the dynamic library and the base HSA API symbols.
  // In static compilation mode this is a no-op.
  memset(out_libhsa, 0, sizeof(*out_libhsa));
  iree_status_t status = iree_hal_amdgpu_libhsa_load_library(
      flags, search_paths, host_allocator, out_libhsa);

  // Initialize HSA. If already loaded this increments the refcount to be paired
  // with the hsa_shut_down we call in deinitialize.
  if (iree_status_is_ok(status)) {
    // ROCR leaks a tremendous amount of global junk.
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_hsa_init(IREE_LIBHSA(out_libhsa));
    IREE_LEAK_CHECK_DISABLE_POP();
    if (iree_status_is_ok(status)) {
      out_libhsa->initialized = true;
    }
  }

  // Load commonly used extensions that we require. These are always dynamic
  // even when statically linking against libhsa as hsa_init must happen first.
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate(
        iree_hsa_system_get_major_extension_table(
            IREE_LIBHSA(out_libhsa), HSA_EXTENSION_AMD_LOADER, 1,
            sizeof(out_libhsa->amd_loader), &out_libhsa->amd_loader),
        IREE_SV("querying HSA_EXTENSION_AMD_LOADER"));
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_libhsa_deinitialize(out_libhsa);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_amdgpu_libhsa_deinitialize(
    iree_hal_amdgpu_libhsa_t* libhsa) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Decrement HSA ref count; others may still have it loaded/in-use.
  if (libhsa->initialized) {
    IREE_LEAK_CHECK_DISABLE_PUSH();
    IREE_IGNORE_ERROR(iree_hsa_shut_down(IREE_LIBHSA(libhsa)));
    IREE_LEAK_CHECK_DISABLE_POP();
  }

  // Unload the dynamic library (this is a ref count).
  iree_hal_amdgpu_libhsa_unload_library(libhsa);

  // Clear all pointers for safety.
  memset(libhsa, 0, sizeof(*libhsa));

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_libhsa_copy(const iree_hal_amdgpu_libhsa_t* libhsa,
                            iree_hal_amdgpu_libhsa_t* out_libhsa) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_libhsa);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try first to bump the HSA ref count. This is very unlikely to fail but we
  // pass it through out of an abundance of caution. The target libhsa_t will
  // call hsa_shut_down to balance the ref count.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hsa_init(IREE_LIBHSA(libhsa)));

  // Copy over/retain the dynamic library data.
  iree_hal_amdgpu_libhsa_copy_library(libhsa, out_libhsa);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_status_from_hsa_status(
    const char* file, const uint32_t line, hsa_status_t hsa_status,
    const char* symbol, const char* message) {
  if (hsa_status == HSA_STATUS_SUCCESS) return iree_ok_status();
  return iree_make_status_with_location(
      file, line, iree_hsa_status_code(hsa_status),
      message ? "[%s] %s: %s; %s" : "[%s] %s: %s", symbol,
      iree_hsa_status_string(hsa_status),
      iree_hsa_status_description(hsa_status), message);
}

//===----------------------------------------------------------------------===//
// HSA API Wrappers
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_LIBHSA_STATIC
#define IREE_HAL_AMDGPU_LIBHSA_LIBPTR(libhsa)
#else
#define IREE_HAL_AMDGPU_LIBHSA_LIBPTR(libhsa) (libhsa)->
#endif  // IREE_HAL_AMDGPU_LIBHSA_STATIC

#if IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE & \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_ALWAYS
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_ALWAYS \
  IREE_TRACE_ZONE_BEGIN
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_TRACE_ALWAYS IREE_TRACE_ZONE_END
#endif  // IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_ALWAYS

#if IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE & \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_SIGNALS
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_SIGNALS \
  IREE_TRACE_ZONE_BEGIN
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_TRACE_SIGNALS IREE_TRACE_ZONE_END
#endif  // IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_SIGNALS

#if IREE_HAL_AMDGPU_LIBHSA_TRACING_MODE & \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_QUEUES
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_QUEUES \
  IREE_TRACE_ZONE_BEGIN
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_TRACE_QUEUES IREE_TRACE_ZONE_END
#endif  // IREE_HAL_AMDGPU_LIBHSA_TRACE_CATEGORY_QUEUES

#if !defined(IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_ALWAYS)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_ALWAYS(...)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_TRACE_ALWAYS(...)
#endif  // !IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_ALWAYS
#if !defined(IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_SIGNALS)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_SIGNALS(...)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_TRACE_SIGNALS(...)
#endif  // !IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_SIGNALS
#if !defined(IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_QUEUES)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_QUEUES(...)
#define IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_TRACE_QUEUES(...)
#endif  // !IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_TRACE_QUEUES

// A thunk into the HSA library for a particular symbol.
// We have thunks so that we can get tracing and status messages with nice
// names for free.
//
// If we wanted to statically link HSA we could use this mechanism to re-route
// all the iree_hsa_* methods directly to the HSA functions.
#define IREE_HAL_AMDGPU_LIBHSA_PFN_hsa_status_t(trace_category, result_type,  \
                                                symbol, decl, args)           \
  iree_status_t iree_##symbol(                                                \
      const iree_hal_amdgpu_libhsa_t* IREE_RESTRICT libhsa, const char* file, \
      const uint32_t line _COMMA_DECL(decl)) {                                \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_##trace_category(z0);             \
                                                                              \
    hsa_status_t hsa_status =                                                 \
        IREE_HAL_AMDGPU_LIBHSA_LIBPTR(libhsa) symbol(args);                   \
                                                                              \
    iree_status_t iree_status = iree_ok_status();                             \
    if (IREE_UNLIKELY(hsa_status != HSA_STATUS_SUCCESS &&                     \
                      hsa_status != HSA_STATUS_INFO_BREAK)) {                 \
      iree_status = iree_status_allocate_f(                                   \
          iree_hsa_status_code(hsa_status), file, line, "[%s] %s: %s",        \
          #symbol, iree_hsa_status_string(hsa_status),                        \
          iree_hsa_status_description(hsa_status));                           \
    }                                                                         \
                                                                              \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_##trace_category(z0);               \
    return iree_status;                                                       \
  }

// A thunk into the HSA library for a particular symbol that does not return a
// value. These are usually high-frequency operations like signals.
#define IREE_HAL_AMDGPU_LIBHSA_PFN_void(trace_category, result_type, symbol, \
                                        decl, args)                          \
  void iree_##symbol(const iree_hal_amdgpu_libhsa_t* IREE_RESTRICT libhsa,   \
                     const char* file,                                       \
                     const uint32_t line _COMMA_DECL(decl)) {                \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_##trace_category(z0);            \
                                                                             \
    IREE_HAL_AMDGPU_LIBHSA_LIBPTR(libhsa) symbol(args);                      \
                                                                             \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_##trace_category(z0);              \
  }

// A thunk into the HSA library for a particular symbol that does not return a
// status value. These are usually high-frequency operations like signals.
#define IREE_HAL_AMDGPU_LIBHSA_PFN_primitive(trace_category, result_type,     \
                                             symbol, decl, args)              \
  result_type iree_##symbol(                                                  \
      const iree_hal_amdgpu_libhsa_t* IREE_RESTRICT libhsa, const char* file, \
      const uint32_t line _COMMA_DECL(decl)) {                                \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_BEGIN_##trace_category(z0);             \
                                                                              \
    result_type _value = IREE_HAL_AMDGPU_LIBHSA_LIBPTR(libhsa) symbol(args);  \
                                                                              \
    IREE_HAL_AMDGPU_LIBHSA_TRACE_ZONE_END_##trace_category(z0);               \
    return _value;                                                            \
  }
#define IREE_HAL_AMDGPU_LIBHSA_PFN_uint32_t IREE_HAL_AMDGPU_LIBHSA_PFN_primitive
#define IREE_HAL_AMDGPU_LIBHSA_PFN_uint64_t IREE_HAL_AMDGPU_LIBHSA_PFN_primitive
#define IREE_HAL_AMDGPU_LIBHSA_PFN_hsa_signal_value_t \
  IREE_HAL_AMDGPU_LIBHSA_PFN_primitive

#define IREE_HAL_AMDGPU_LIBHSA_PFN(trace_category, result_type, symbol, decl, \
                                   args)                                      \
  IREE_HAL_AMDGPU_LIBHSA_PFN_##result_type(trace_category, result_type,       \
                                           symbol, DECL(decl), ARGS(args))

#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#define _COMMA_DECL(...) __VA_OPT__(, ) __VA_ARGS__
#include "iree/hal/drivers/amdgpu/util/libhsa_tables.h"  // IWYU pragma: export
#undef _COMMA_DECL
