// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/system_library_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/executable_library_util.h"
#include "iree/hal/local/executable_plugin_manager.h"
#include "iree/hal/local/local_executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_system_executable_footer_t
//===----------------------------------------------------------------------===//

// An optional footer that may exist on the system library that is used to add
// additional debug information for use directly by IREE, such as PDB or dSYM
// files. This is only expected to be present when there is a debug database
// but we may want to extend it in the future.
typedef struct iree_hal_system_executable_footer_t {
  uint8_t magic[8];  // IREE_HAL_SYSTEM_EXECUTABLE_FOOTER_MAGIC
  uint32_t version;  // IREE_HAL_SYSTEM_EXECUTABLE_FOOTER_VERSION
  uint32_t flags;    // reserved
  // Offset of the library within the parent data stream.
  // Almost always zero but here in case we want to allow for chaining.
  uint64_t library_offset;
  // Size of the system library in bytes.
  uint64_t library_size;
  // Offset of the start of the embedded debug database within the parent data
  // stream. There may be padding between the library and this offset.
  uint64_t debug_offset;
  // Size of the debug database in bytes.
  uint64_t debug_size;
} iree_hal_system_executable_footer_t;

// EXPERIMENTAL: this is not a stable interface yet. The binary format may
// change at any time.
#define IREE_HAL_SYSTEM_EXECUTABLE_FOOTER_MAGIC "IREEDBG\0"
#define IREE_HAL_SYSTEM_EXECUTABLE_FOOTER_VERSION 0

// Tries to find an iree_hal_system_executable_footer_t at the end of the
// given executable data stream.
static const iree_hal_system_executable_footer_t*
iree_hal_system_executable_try_query_footer(
    iree_const_byte_span_t executable_data) {
  if (executable_data.data_length <
      sizeof(iree_hal_system_executable_footer_t)) {
    return NULL;
  }
  const uint8_t* footer_ptr = executable_data.data +
                              executable_data.data_length -
                              sizeof(iree_hal_system_executable_footer_t);
  const iree_hal_system_executable_footer_t* footer =
      (const iree_hal_system_executable_footer_t*)(footer_ptr);
  static_assert(sizeof(IREE_HAL_SYSTEM_EXECUTABLE_FOOTER_MAGIC) - /*NUL*/ 1 ==
                    sizeof(footer->magic),
                "magic number value must match struct size");
  if (memcmp(footer->magic, IREE_HAL_SYSTEM_EXECUTABLE_FOOTER_MAGIC,
             sizeof(footer->magic)) != 0) {
    return NULL;
  }
  return footer;
}

//===----------------------------------------------------------------------===//
// iree_hal_system_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_system_executable_t {
  iree_hal_local_executable_t base;

  // Loaded platform dynamic library.
  iree_dynamic_library_t* handle;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

#if (IREE_TRACING_FEATURES)
  // Optional mapping of filenames to custom contents.
  tracy_file_mapping* file_mapping;
#endif

  // Queried metadata from the library.
  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;

  iree_hal_pipeline_layout_t* layouts[];
} iree_hal_system_executable_t;

static const iree_hal_local_executable_vtable_t
    iree_hal_system_executable_vtable;

// Loads the executable and optional debug database from the given
// |executable_data| in memory. The memory must remain live for the lifetime
// of the executable.
static iree_status_t iree_hal_system_executable_load(
    iree_hal_system_executable_t* executable,
    iree_const_byte_span_t executable_data, iree_allocator_t host_allocator) {
  // Check to see if the library has a footer indicating embedded debug data.
  iree_const_byte_span_t library_data = iree_make_const_byte_span(NULL, 0);
  iree_const_byte_span_t debug_data = iree_make_const_byte_span(NULL, 0);
  const iree_hal_system_executable_footer_t* footer =
      iree_hal_system_executable_try_query_footer(executable_data);
  if (footer) {
    // Debug file present; split the data contents.
    iree_host_size_t data_length =
        executable_data.data_length - sizeof(*footer);
    if (footer->library_size > data_length ||
        footer->debug_offset + footer->debug_size > data_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "system library footer references out of range bytes");
    }
    library_data =
        iree_make_const_byte_span(executable_data.data, footer->library_size);
    debug_data = iree_make_const_byte_span(
        executable_data.data + footer->debug_offset, footer->debug_size);
  } else {
    // Entire data contents are the library.
    library_data = executable_data;
  }

  IREE_RETURN_IF_ERROR(iree_dynamic_library_load_from_memory(
      iree_make_cstring_view("aot"), library_data,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &executable->handle));

  if (debug_data.data_length > 0) {
    IREE_RETURN_IF_ERROR(iree_dynamic_library_attach_symbols_from_memory(
        executable->handle, debug_data));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_system_executable_query_library(
    iree_hal_system_executable_t* executable) {
  // Get the exported symbol used to get the library metadata.
  iree_hal_executable_library_query_fn_t query_fn = NULL;
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(
      executable->handle, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME,
      (void**)&query_fn));

  // Query for a compatible version of the library.
  executable->library.header =
      query_fn(IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
               &executable->base.environment);
  if (!executable->library.header) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable does not support this version of the runtime (%08X)",
        IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
  }
  const iree_hal_executable_library_header_t* header =
      *executable->library.header;

  // Ensure that if the library is built for a particular sanitizer that we also
  // were compiled with that sanitizer enabled.
  switch (header->sanitizer) {
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE:
      // Always safe even if the host has a sanitizer enabled; it just means
      // that we won't be able to catch anything from within the executable,
      // however checks outside will (often) still trigger when guard pages are
      // dirtied/etc.
      break;
#if defined(IREE_SANITIZER_ADDRESS)
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS:
      // ASAN is compiled into the host and we can load this library.
      break;
#else
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "executable library is compiled with ASAN support but the host "
          "runtime is not compiled with it enabled; add -fsanitize=address to "
          "the runtime compilation options");
#endif  // IREE_SANITIZER_ADDRESS
#if defined(IREE_SANITIZER_THREAD)
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_THREAD:
      // TSAN is compiled into the host and we can load this library.
      break;
#else
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_THREAD:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "executable library is compiled with TSAN support but the host "
          "runtime is not compiled with it enabled; add -fsanitize=thread to "
          "the runtime compilation options");
#endif  // IREE_SANITIZER_THREAD
    default:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "executable library requires a sanitizer the host runtime is not "
          "compiled to enable/understand: %u",
          (uint32_t)header->sanitizer);
  }

  executable->identifier = iree_make_cstring_view(header->name);
  executable->base.dispatch_attrs = executable->library.v0->exports.attrs;
  return iree_ok_status();
}

static int iree_hal_system_executable_import_thunk_v0(
    iree_hal_executable_import_v0_t fn_ptr, void* params, void* context,
    void* reserved) {
  return fn_ptr(params, context, reserved);
}

static iree_status_t iree_hal_system_executable_create(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  printf("iree_hal_system_executable_create()\n");
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(executable_params->executable_data.data &&
                       executable_params->executable_data.data_length);
  IREE_ASSERT_ARGUMENT(!executable_params->pipeline_layout_count ||
                       executable_params->pipeline_layouts);
  IREE_ASSERT_ARGUMENT(!executable_params->constant_count ||
                       executable_params->constants);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_system_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_params->pipeline_layout_count * sizeof(*executable->layouts) +
      executable_params->constant_count * sizeof(*executable_params->constants);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_initialize(
        &iree_hal_system_executable_vtable,
        executable_params->pipeline_layout_count,
        executable_params->pipeline_layouts, &executable->layouts[0],
        host_allocator, &executable->base);
  }

  // Copy executable constants so we own them.
  if (iree_status_is_ok(status) && executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + sizeof(*executable) +
                    executable_params->pipeline_layout_count *
                        sizeof(*executable->layouts));
    memcpy(target_constants, executable_params->constants,
           executable_params->constant_count *
               sizeof(*executable_params->constants));
    executable->base.environment.constants = target_constants;
  }

  // Attempt to extract the embedded library and load it.
  if (iree_status_is_ok(status)) {
    status = iree_hal_system_executable_load(
        executable, executable_params->executable_data, host_allocator);
  }

  // Query metadata and get the entry point function pointers.
  if (iree_status_is_ok(status)) {
    status = iree_hal_system_executable_query_library(executable);
  }

  // Resolve imports, if any.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_initialize_imports(
        &executable->base.environment, import_provider,
        &executable->library.v0->imports,
        iree_hal_system_executable_import_thunk_v0, host_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = IREE_HAL_EXECUTABLE_LIBRARY_SETUP_TRACING(
        executable->library.v0, host_allocator, &executable->file_mapping);
  }

  // Verify that the library matches the executable params.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_verify(executable_params,
                                                executable->library.v0);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(executable->handle);

  iree_hal_executable_library_deinitialize_imports(
      &executable->base.environment, host_allocator);

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_system_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;
  const iree_hal_executable_library_v0_t* library = executable->library.v0;
  if (IREE_UNLIKELY(ordinal >= library->exports.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }
  // printf("iree_hal_system_executable_issue_call(): ordinal %zu\n", ordinal);
  // printf("iree_hal_system_executable_issue_call(): identifier %s\n",
  // executable->identifier.data);

  IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN(z0, executable->identifier,
                                                    library, ordinal);
  int ret = library->exports.ptrs[ordinal](&base_executable->environment,
                                           dispatch_state, workgroup_state);
  IREE_TRACE_ZONE_END(z0);

  return ret == 0 ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_INTERNAL,
                        "executable entry point returned catastrophic error %d",
                        ret);
}

static const iree_hal_local_executable_vtable_t
    iree_hal_system_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_system_executable_destroy,
            },
        .issue_call = iree_hal_system_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_system_library_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_system_library_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  iree_hal_executable_plugin_manager_t* plugin_manager;
} iree_hal_system_library_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable;

iree_status_t iree_hal_system_library_loader_create(
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_system_library_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_system_library_loader_vtable,
        iree_hal_executable_plugin_manager_provider(plugin_manager),
        &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    executable_loader->plugin_manager = plugin_manager;
    iree_hal_executable_plugin_manager_retain(
        executable_loader->plugin_manager);
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_library_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_system_library_loader_t* executable_loader =
      (iree_hal_system_library_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_plugin_manager_release(executable_loader->plugin_manager);
  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

#if defined(IREE_PLATFORM_APPLE)
#define IREE_PLATFORM_DYLIB_TYPE "dylib"
#elif defined(IREE_PLATFORM_WINDOWS)
#define IREE_PLATFORM_DYLIB_TYPE "dll"
#elif defined(IREE_PLATFORM_EMSCRIPTEN)
#define IREE_PLATFORM_DYLIB_TYPE "wasm"
#else
#define IREE_PLATFORM_DYLIB_TYPE "elf"
#endif  // IREE_PLATFORM_*

static bool iree_hal_system_library_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_starts_with(
      executable_format,
      iree_make_cstring_view("system-" IREE_PLATFORM_DYLIB_TYPE "-" IREE_ARCH));
}

static iree_status_t iree_hal_system_library_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  iree_hal_system_library_loader_t* executable_loader =
      (iree_hal_system_library_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform the load (and requisite disgusting hackery).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_system_executable_create(
              executable_params, base_executable_loader->import_provider,
              executable_loader->host_allocator, out_executable));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable = {
        .destroy = iree_hal_system_library_loader_destroy,
        .query_support = iree_hal_system_library_loader_query_support,
        .try_load = iree_hal_system_library_loader_try_load,
};
