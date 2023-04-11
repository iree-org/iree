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
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_EMSCRIPTEN)

#include <dlfcn.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct iree_dynamic_library_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  // dlopen shared object handle.
  void* handle;
};

// Allocate a new string from |allocator| returned in |out_file_path| containing
// a path to a unique file on the filesystem.
static iree_status_t iree_dynamic_library_make_temp_file_path(
    const char* prefix, const char* extension, iree_allocator_t allocator,
    const char* tmpdir, char** out_file_path) {
  // Stamp in a unique file name (replacing XXXXXX in the string).
  char temp_path[512];
  if (snprintf(temp_path, sizeof(temp_path), "%s/iree_dylib_XXXXXX", tmpdir) >=
      sizeof(temp_path)) {
    // NOTE: we could dynamically allocate things, but didn't seem worth it.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "TMPDIR name too long (>%zu chars); keep it reasonable",
        sizeof(temp_path));
  }
  int fd = mkstemp(temp_path);
  if (fd < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "unable to mkstemp file");
  }

  // Allocate storage for the full file path and format it in.
  int file_path_length =
      snprintf(NULL, 0, "%s_%s.%s", temp_path, prefix, extension);
  if (file_path_length < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unable to form temp path string");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, file_path_length + /*NUL=*/1, (void**)out_file_path));
  snprintf(*out_file_path, file_path_length + /*NUL=*/1, "%s_%s.%s", temp_path,
           prefix, extension);

  // Canonicalize away any double path separators.
  iree_file_path_canonicalize(*out_file_path, file_path_length);

  return iree_ok_status();
}

// Creates a temp file and writes the |source_data| into it.
// The file path is returned in |out_file_path|.
static iree_status_t iree_dynamic_library_write_temp_file(
    iree_const_byte_span_t source_data, const char* prefix,
    const char* extension, iree_allocator_t allocator, const char* tmpdir,
    char** out_file_path) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reserve a temp file path we can write to.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_make_temp_file_path(prefix, extension, allocator,
                                                   tmpdir, out_file_path));

  iree_status_t status = iree_ok_status();

  // Open the file for writing.
  FILE* file_handle = fopen(*out_file_path, "wb");
  if (file_handle == NULL) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "unable to open file '%s'", *out_file_path);
  }

  // Write all file bytes.
  if (iree_status_is_ok(status)) {
    if (fwrite((char*)source_data.data, source_data.data_length, 1,
               file_handle) != 1) {
      status =
          iree_make_status(iree_status_code_from_errno(errno),
                           "unable to write file span of %zu bytes to '%s'",
                           source_data.data_length, *out_file_path);
    }
  }

  if (file_handle != NULL) {
    fclose(file_handle);
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
    void* handle, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  *out_library = NULL;

  iree_dynamic_library_t* library = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*library), (void**)&library));
  memset(library, 0, sizeof(*library));
  iree_atomic_ref_count_init(&library->ref_count);
  library->allocator = allocator;
  library->handle = handle;

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
  void* handle = NULL;
  iree_host_size_t i = 0;
  for (i = 0; i < search_path_count; ++i) {
    handle = dlopen(search_paths[i], RTLD_LAZY | RTLD_LOCAL);
    if (handle) break;
  }
  if (!handle) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "failed to load dynamic library (possibly not "
                            "found on any search path): %s",
                            dlerror());
  }

  iree_dynamic_library_t* library = NULL;
  iree_status_t status =
      iree_dynamic_library_create(handle, allocator, &library);

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    dlclose(handle);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_once_flag iree_dynamic_library_temp_dir_init_once_flag_ =
    IREE_ONCE_FLAG_INIT;
static const char* iree_dynamic_library_temp_dir_path_;
static bool iree_dynamic_library_temp_dir_valid_;
static bool iree_dynamic_library_temp_dir_preserve_;

static bool iree_dynamic_library_path_is_null_or_empty(const char* path) {
  return path == NULL || path[0] == 0;
}

static void iree_dynamic_library_init_temp_dir(void) {
  // Semantics of IREE_PRESERVE_DYLIB_TEMP_FILES:
  // * If the environment variable is not set, temp files are not preserved.
  // * If the environment variable is set to "1", temp files are preserved to
  //   some default temp directory. The TMPDIR environment variable is used if
  //   set, otherwise a hardcoded default path is used. Example:
  //     $ IREE_PRESERVE_DYLIB_TEMP_FILES=1 iree-run-module ...
  // * If the environment variable is set to any other string than "1", temp
  // files
  //   are preserved, and the value of the environment variable is interpreted
  //   as the path of the temporary directory to use. Example:
  //     $ IREE_PRESERVE_DYLIB_TEMP_FILES=/tmp/iree-benchmarks iree-run-module
  //     ...
  const char* path = getenv("IREE_PRESERVE_DYLIB_TEMP_FILES");
  bool preserve = !iree_dynamic_library_path_is_null_or_empty(path);
  if (!path || !strcmp(path, "1")) {
    // TMPDIR is a unix semi-standard thing. It's even defined by default on
    // Android for the regular shell user (but not root).
    path = getenv("TMPDIR");
    if (iree_dynamic_library_path_is_null_or_empty(path)) {
#ifdef __ANDROID__
      path = "/data/local/tmp";
#else
      path = "/tmp";
#endif  // __ANDROID__
    }
  }
  iree_dynamic_library_temp_dir_path_ = path;
  iree_dynamic_library_temp_dir_preserve_ = preserve;
  // Validate that temp_dir it is the path of a directory. Could fail if it was
  // user-provided, or on an Android device where /data/local/tmp hasn't been
  // created yet.
  struct stat s;
  iree_dynamic_library_temp_dir_valid_ =
      stat(path, &s) == 0 && (s.st_mode & S_IFMT) == S_IFDIR;
}

// TODO(#3845): use dlopen on an fd with either dlopen(/proc/self/fd/NN),
// fdlopen, or android_dlopen_ext to avoid needing to write the file to disk.
// Can fallback to memfd_create + dlopen where available, and fallback from
// that to disk (maybe just windows/mac).
iree_status_t iree_dynamic_library_load_from_memory(
    iree_string_view_t identifier, iree_const_byte_span_t buffer,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  iree_call_once(&iree_dynamic_library_temp_dir_init_once_flag_,
                 iree_dynamic_library_init_temp_dir);

  if (!iree_dynamic_library_temp_dir_valid_) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "path of dylib temp files (%s) is not the path of a directory",
        iree_dynamic_library_temp_dir_path_);
  }

  // Extract the library to a temp file.
  char* temp_path = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_dynamic_library_write_temp_file(
              buffer, "mem_", "so", allocator,
              iree_dynamic_library_temp_dir_path_, &temp_path));

  // Load using the normal load from file routine.
  iree_status_t status = iree_dynamic_library_load_from_file(
      temp_path, flags, allocator, out_library);

  // Unlink the temp file - it's still open by the loader but won't be
  // accessible to anyone else and will be deleted once the library is
  // unloaded. Note that we don't remove the file if the user requested we keep
  // it around for tooling to access.
  if (!iree_dynamic_library_temp_dir_preserve_) {
    remove(temp_path);
  }
  iree_allocator_free(allocator, temp_path);

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
  if (library->handle != NULL) {
    dlclose(library->handle);
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

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
  void* fn = dlsym(library->handle, symbol_name);
  if (!fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "symbol '%s' not found in library", symbol_name);
  }
  *out_fn = fn;
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path) {
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer) {
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_*
