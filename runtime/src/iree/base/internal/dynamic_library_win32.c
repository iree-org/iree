// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/call_once.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"

#if defined(IREE_PLATFORM_WINDOWS)

// TODO(benvanik): support PDB overlays when tracy is not enabled; we'll
// need to rearrange how the dbghelp lock is handled for that (probably moving
// it here and having the tracy code redirect to this).
#if defined(TRACY_ENABLE)
#define IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT 1
#pragma warning(disable : 4091)
#include <dbghelp.h>
void IREEDbgHelpLock(void);
void IREEDbgHelpUnlock(void);
#endif  // TRACY_ENABLE

struct iree_dynamic_library_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  // Base module name used as an identifier. When loaded from a file this must
  // be the basename for dbghelp to be able to find symbols.
  // Owned and allocated as part of the struct upon creation.
  // Has NUL terminator for compatibility with Windows APIs.
  char* identifier;

  // File path of the loaded module, if loaded from one.
  // Owned and allocated as part of the struct upon creation.
  // Has NUL terminator for compatibility with Windows APIs.
  char* module_path;

  // Windows module handle.
  HMODULE module;

  // 0 or more file paths that were created as part of the loading of the
  // library or attaching of symbols from memory.
  //
  // Each path string is allocated using the |allocator| and freed during
  // library deletion.
  iree_host_size_t temp_file_count;
  char* temp_file_paths[2];
};

static iree_once_flag iree_dynamic_library_temp_path_flag_ =
    IREE_ONCE_FLAG_INIT;
static char iree_dynamic_library_temp_path_base_[MAX_PATH + 1];
static void iree_dynamic_library_init_temp_paths(void) {
  // Query the temp path from the OS. This can be overridden with the following
  // environment variables: [TMP, TEMP, USERPROFILE].
  //
  // See:
  // https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-gettemppatha
  char temp_path[MAX_PATH];
  DWORD temp_path_length = GetTempPathA(IREE_ARRAYSIZE(temp_path), temp_path);

  // Append the process ID to the path; this is like what _mktemp does but
  // without all the hoops.
  snprintf(iree_dynamic_library_temp_path_base_,
           sizeof(iree_dynamic_library_temp_path_base_), "%s\\iree_dylib_%08X",
           temp_path, GetCurrentProcessId());

  // Canonicalize away any double path separators.
  iree_file_path_canonicalize(iree_dynamic_library_temp_path_base_,
                              strlen(iree_dynamic_library_temp_path_base_));
}

// Allocate a new string from |allocator| returned in |out_file_path| containing
// a path to a unique file on the filesystem.
static iree_status_t iree_dynamic_library_make_temp_file_path(
    const char* prefix, const char* extension, iree_allocator_t allocator,
    char** out_file_path) {
  // Ensure the root temp paths are queried/initialized.
  iree_call_once(&iree_dynamic_library_temp_path_flag_,
                 iree_dynamic_library_init_temp_paths);

  // Generate a per-file unique identifier only unique **within** the current
  // process. We combine this with the _mktemp path that should be unique to the
  // process itself.
  static iree_atomic_int32_t next_unique_id = IREE_ATOMIC_VAR_INIT(0);
  // relaxed because we only care about uniqueness, we don't care about ordering
  // of accesses to unique_id w.r.t. other memory operations.
  uint32_t unique_id = (uint32_t)iree_atomic_fetch_add_int32(
      &next_unique_id, 1, iree_memory_order_relaxed);

  // Allocate storage for the full file path and format it in.
  int file_path_length =
      snprintf(NULL, 0, "%s_%s_%08X.%s", iree_dynamic_library_temp_path_base_,
               prefix, unique_id, extension);
  if (file_path_length < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unable to form temp path string");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, file_path_length + /*NUL=*/1, (void**)out_file_path));
  snprintf(*out_file_path, file_path_length + /*NUL=*/1, "%s_%s_%08X.%s",
           iree_dynamic_library_temp_path_base_, prefix, unique_id, extension);

  return iree_ok_status();
}

// Creates a temp file and writes the |source_data| into it.
// The file path is returned in |out_file_path|.
static iree_status_t iree_dynamic_library_write_temp_file(
    iree_const_byte_span_t source_data, const char* prefix,
    const char* extension, iree_allocator_t allocator, char** out_file_path) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reserve a temp file path we can write to.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_make_temp_file_path(prefix, extension, allocator,
                                                   out_file_path));

  iree_status_t status = iree_ok_status();

  // Open the file for writing.
  HANDLE file_handle = CreateFileA(
      /*lpFileName=*/*out_file_path, /*dwDesiredAccess=*/GENERIC_WRITE,
      /*dwShareMode=*/FILE_SHARE_DELETE, /*lpSecurityAttributes=*/NULL,
      /*dwCreationDisposition=*/CREATE_ALWAYS,
      /*dwFlagsAndAttributes=*/FILE_ATTRIBUTE_TEMPORARY,
      /*hTemplateFile=*/NULL);
  if (file_handle == INVALID_HANDLE_VALUE) {
    status = iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                              "unable to open file '%s'", *out_file_path);
  }

  // Write all file bytes.
  if (iree_status_is_ok(status)) {
    if (WriteFile(file_handle, source_data.data, (DWORD)source_data.data_length,
                  NULL, NULL) == FALSE) {
      status = iree_make_status(
          iree_status_code_from_win32_error(GetLastError()),
          "unable to write file span of %" PRIhsz " bytes to '%s'",
          source_data.data_length, *out_file_path);
    }
  }

  if (file_handle != NULL) {
    CloseHandle(file_handle);
    file_handle = NULL;
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, *out_file_path);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Allocates an iree_dynamic_library_t with the given allocator.
static iree_status_t iree_dynamic_library_create(
    iree_string_view_t identifier, iree_string_view_t module_path,
    HMODULE module, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  *out_library = NULL;

  iree_dynamic_library_t* library = NULL;
  iree_host_size_t total_size =
      sizeof(*library) + (identifier.size + 1) + (module_path.size + 1);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&library));
  memset(library, 0, total_size);
  iree_atomic_ref_count_init(&library->ref_count);
  library->allocator = allocator;
  library->module = module;

  library->identifier = (char*)library + sizeof(*library);
  memcpy(library->identifier, identifier.data, identifier.size);
  library->identifier[identifier.size] = 0;  // NUL

  library->module_path = library->identifier + (identifier.size + 1);
  memcpy(library->module_path, module_path.data, module_path.size);
  library->module_path[module_path.size] = 0;  // NUL

  *out_library = library;
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_load_from_file(
    const char* file_path, iree_dynamic_library_flags_t flags,
    iree_allocator_t allocator, iree_dynamic_library_t** out_library) {
  return iree_dynamic_library_load_from_files(1, &file_path, flags, allocator,
                                              out_library);
}

iree_status_t iree_dynamic_library_load_from_files(
    iree_host_size_t search_path_count, const char* const* search_paths,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  // Try to load the module from the set of search paths provided.
  HMODULE module = NULL;
  iree_host_size_t i = 0;
  for (i = 0; i < search_path_count; ++i) {
    module = LoadLibraryA(search_paths[i]);
    if (module) break;
  }
  if (!module) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "dynamic library not found on any search path");
  }

  iree_string_view_t file_path = iree_make_cstring_view(search_paths[i]);
  iree_string_view_t identifier = iree_file_path_basename(file_path);

  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_dynamic_library_create(
      identifier, file_path, module, allocator, &library);

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    FreeLibrary(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_dynamic_library_load_from_memory(
    iree_string_view_t identifier, iree_const_byte_span_t buffer,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  // Extract the library to a temp file.
  char* temp_path = NULL;
  iree_status_t status = iree_dynamic_library_write_temp_file(
      buffer, "mem", "dll", allocator, &temp_path);

  if (iree_status_is_ok(status)) {
    // Load using the normal load from file routine.
    status = iree_dynamic_library_load_from_file(temp_path, flags, allocator,
                                                 out_library);
  }
  if (iree_status_is_ok(status)) {
    // Associate the temp path to the library; the temp_path string and the
    // backing file will be deleted when the library is closed.
    iree_dynamic_library_t* library = *out_library;
    library->temp_file_paths[library->temp_file_count++] = temp_path;
  } else {
    iree_allocator_free(allocator, temp_path);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_dynamic_library_delete(iree_dynamic_library_t* library) {
  iree_allocator_t allocator = library->allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  // Leak the library when tracing, since the profiler may still be reading it.
  // TODO(benvanik): move to an atexit handler instead, verify with ASAN/MSAN
  // TODO(scotttodd): Make this compatible with testing:
  //     two test cases, one for each function in the same executable
  //     first test case passes, second fails to open the file (already open)
#else
  // Close the library first as it may be loaded from one of the temp files we
  // are about to delete.
  if (library->module != NULL) {
    FreeLibrary(library->module);
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  // Cleanup all temp files.
  for (iree_host_size_t i = 0; i < library->temp_file_count; ++i) {
    char* file_path = library->temp_file_paths[i];
    DeleteFileA(file_path);
    iree_allocator_free(allocator, file_path);
  }

  iree_allocator_free(allocator, library);

  IREE_TRACE_ZONE_END(z0);
}

void iree_dynamic_library_retain(iree_dynamic_library_t* library) {
  if (library) {
    iree_atomic_ref_count_inc(&library->ref_count);
  }
}

void iree_dynamic_library_release(iree_dynamic_library_t* library) {
  if (library && iree_atomic_ref_count_dec(&library->ref_count) == 1) {
    iree_dynamic_library_delete(library);
  }
}

iree_status_t iree_dynamic_library_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(symbol_name);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = NULL;
  void* fn = GetProcAddress(library->module, symbol_name);
  if (!fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "symbol '%s' not found in library", symbol_name);
  }
  *out_fn = fn;
  return iree_ok_status();
}

#if defined(IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT)

typedef struct {
  const char* module_path;
  DWORD64 module_base;
  ULONG module_size;
} ModuleEnumCallbackState;

static BOOL EnumLoadedModulesCallback(PCSTR ModuleName, DWORD64 ModuleBase,
                                      ULONG ModuleSize, PVOID UserContext) {
  ModuleEnumCallbackState* state = (ModuleEnumCallbackState*)UserContext;
  if (strcmp(ModuleName, state->module_path) != 0) {
    return TRUE;  // not a match; continue
  }
  state->module_base = ModuleBase;
  state->module_size = ModuleSize;
  return FALSE;  // match found; stop enumeration
}

iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREEDbgHelpLock();

  // Useful for debugging this logic; will print search paths and results:
  // SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEBUG);

  // Enumerates all loaded modules in the process to extract the module
  // base/size parameters we need to overlay the PDB. There's other ways to
  // get this (such as registering a LdrDllNotification callback and snooping
  // the values during LoadLibrary or using CreateToolhelp32Snapshot), however
  // EnumerateLoadedModules is in dbghelp which we are using anyway.
  ModuleEnumCallbackState state;
  memset(&state, 0, sizeof(state));
  state.module_path = library->module_path;
  EnumerateLoadedModules64(GetCurrentProcess(), EnumLoadedModulesCallback,
                           &state);

  // Load the PDB file and overlay it onto the already-loaded module at the
  // address range it got loaded into.
  if (state.module_base != 0) {
    SymLoadModuleEx(GetCurrentProcess(), NULL, file_path, library->identifier,
                    state.module_base, state.module_size, NULL, 0);
  }

  IREEDbgHelpUnlock();

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (library->temp_file_count + 1 > IREE_ARRAYSIZE(library->temp_file_paths)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many temp files attached");
  }

  // Extract the library to a temp file.
  char* temp_path = NULL;
  iree_status_t status = iree_dynamic_library_write_temp_file(
      buffer, "mem", "pdb", library->allocator, &temp_path);
  if (iree_status_is_ok(status)) {
    // Associate the temp path to the library; the temp_path string and the
    // backing file will be deleted when the library is closed.
    library->temp_file_paths[library->temp_file_count++] = temp_path;

    // Attempt to attach the extracted temp file to the module.
    status = iree_dynamic_library_attach_symbols_from_file(library, temp_path);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path) {
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer) {
  return iree_ok_status();
}

#endif  // IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT

#endif  // IREE_PLATFORM_WINDOWS
