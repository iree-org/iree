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

#include "iree/hal/local/loaders/legacy_library_loader.h"

#include "iree/base/dynamic_library.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/file_path.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/local_executable.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/dylib_executable_def_reader.h"
#include "iree/schemas/dylib_executable_def_verifier.h"

//===----------------------------------------------------------------------===//
// Verification and file utilities
//===----------------------------------------------------------------------===//

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_dylib_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    iree_host_size_t expected_entry_point_count) {
  // Special handling for valid but mismatching flatbuffers.
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16 ||
      !flatbuffers_has_identifier(flatbuffer_data.data,
                                  iree_DyLibExecutableDef_file_identifier)) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_DyLibExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_DyLibExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count != expected_entry_point_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_point_count, expected_entry_point_count);
  }

  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  if (!flatbuffers_uint8_vec_len(
          iree_DyLibExecutableDef_library_embedded_get(executable_def))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable library_embedded is missing/empty");
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_legacy_executable_t
//===----------------------------------------------------------------------===//

typedef void (*iree_hal_legacy_executable_fn_ptr_t)(void* const*,
                                                    const uint32_t*,
                                                    const uint32_t*,
                                                    const uint32_t*,
                                                    const uint32_t*);

typedef struct {
  iree_hal_local_executable_t base;

  // Flatbuffer definition referencing the executable memory.
  iree_DyLibExecutableDef_table_t def;

  // Temporary files created as part of extraction.
  // Strings are allocated from the host allocator.
  iree_host_size_t temp_file_count;
  iree_string_view_t temp_files[8];

  // Loaded platform dynamic library.
  iree::DynamicLibrary* library;

  // Resolved entry points from the dynamic library.
  iree_host_size_t entry_fn_count;
  iree_hal_legacy_executable_fn_ptr_t entry_fns[];
} iree_hal_legacy_executable_t;

extern const iree_hal_local_executable_vtable_t
    iree_hal_legacy_executable_vtable;

static iree_status_t iree_hal_legacy_executable_extract_and_load(
    iree_hal_legacy_executable_t* executable, iree_allocator_t host_allocator) {
  // Write the embedded library out to a temp file, since all of the dynamic
  // library APIs work with files. We could instead use in-memory files on
  // platforms where that is convenient.
  //
  // TODO(#3845): use dlopen on an fd with either dlopen(/proc/self/fd/NN),
  // fdlopen, or android_dlopen_ext to avoid needing to write the file to disk.
  // Can fallback to memfd_create + dlopen where available, and fallback from
  // that to disk (maybe just windows/mac).
  std::string library_temp_path;
  IREE_RETURN_IF_ERROR(
      iree::file_io::GetTempFile("dylib_executable", &library_temp_path));

// Add platform-specific file extensions so opinionated dynamic library
// loaders are more likely to find the file:
#if defined(IREE_PLATFORM_WINDOWS)
  library_temp_path += ".dll";
#else
  library_temp_path += ".so";
#endif  // IREE_PLATFORM_WINDOWS

  iree_string_view_t library_temp_file = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_allocator_clone(host_allocator,
                           iree_make_const_byte_span(library_temp_path.data(),
                                                     library_temp_path.size()),
                           (void**)&library_temp_file.data));
  library_temp_file.size = library_temp_path.size();
  executable->temp_files[executable->temp_file_count++] = library_temp_file;

  flatbuffers_uint8_vec_t embedded_library_vec =
      iree_DyLibExecutableDef_library_embedded_get(executable->def);
  IREE_RETURN_IF_ERROR(iree::file_io::SetFileContents(
      library_temp_path,
      absl::string_view(reinterpret_cast<const char*>(embedded_library_vec),
                        flatbuffers_uint8_vec_len(embedded_library_vec))));

  std::unique_ptr<iree::DynamicLibrary> library;
  IREE_RETURN_IF_ERROR(
      iree::DynamicLibrary::Load(library_temp_path.c_str(), &library));

  flatbuffers_string_t debug_database_filename =
      iree_DyLibExecutableDef_debug_database_filename_get(executable->def);
  flatbuffers_uint8_vec_t debug_database_embedded_vec =
      iree_DyLibExecutableDef_debug_database_embedded_get(executable->def);
  if (flatbuffers_string_len(debug_database_filename) &&
      flatbuffers_uint8_vec_len(debug_database_embedded_vec)) {
    IREE_TRACE_SCOPE0("DyLibExecutable::AttachDebugDatabase");
    auto debug_database_path = iree::file_path::JoinPaths(
        iree::file_path::DirectoryName(library_temp_path),
        absl::string_view(debug_database_filename,
                          flatbuffers_string_len(debug_database_filename)));
    iree_string_view_t debug_database_file = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        host_allocator,
        iree_make_const_byte_span(debug_database_path.data(),
                                  debug_database_path.size()),
        (void**)&debug_database_file.data));
    debug_database_file.size = debug_database_path.size();
    executable->temp_files[executable->temp_file_count++] = debug_database_file;
    IREE_IGNORE_ERROR(iree::file_io::SetFileContents(
        debug_database_path,
        absl::string_view(
            reinterpret_cast<const char*>(debug_database_embedded_vec),
            flatbuffers_uint8_vec_len(debug_database_embedded_vec))));
    library->AttachDebugDatabase(debug_database_path.c_str());
  }

  executable->library = library.release();

  return iree_ok_status();
}

static iree_status_t iree_hal_legacy_executable_resolve_symbols(
    iree_hal_legacy_executable_t* executable) {
  flatbuffers_string_vec_t entry_points_vec =
      iree_DyLibExecutableDef_entry_points_get(executable->def);
  for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
    flatbuffers_string_t entry_point_str =
        flatbuffers_string_vec_at(entry_points_vec, i);
    void* symbol = executable->library->GetSymbol(entry_point_str);
    if (!symbol) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "symbol %s not exported by the dynamic library, check visibility",
          entry_point_str);
    }
    executable->entry_fns[i] = (iree_hal_legacy_executable_fn_ptr_t)symbol;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_legacy_executable_create(
    iree_DyLibExecutableDef_table_t executable_def,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_def);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  flatbuffers_string_vec_t entry_points_vec =
      iree_DyLibExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count != executable_layout_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_point_count, executable_layout_count);
  }

  iree_hal_legacy_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) + entry_point_count * sizeof(*executable->entry_fns) +
      executable_layout_count * sizeof(iree_hal_local_executable_layout_t);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_layout_t** executable_layouts_ptr =
        (iree_hal_local_executable_layout_t**)(((uint8_t*)executable) +
                                               sizeof(*executable) +
                                               entry_point_count *
                                                   sizeof(
                                                       *executable->entry_fns));
    iree_hal_local_executable_initialize(
        &iree_hal_legacy_executable_vtable, executable_layout_count,
        executable_layouts, executable_layouts_ptr, host_allocator,
        &executable->base);
    executable->def = executable_def;
    executable->entry_fn_count = entry_point_count;
  }
  if (iree_status_is_ok(status)) {
    // Attempt to extract the embedded flatbuffer library and load it.
    // Will scribble information into executable.
    // This is bad, but ehh all this is getting deleted soon and hopefully we
    // can avoid ever touching the disk at all.
    status =
        iree_hal_legacy_executable_extract_and_load(executable, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    // Attempt to resolve symbols for all entry points.
    status = iree_hal_legacy_executable_resolve_symbols(executable);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_legacy_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_legacy_executable_t* executable =
      (iree_hal_legacy_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  // Leak the library when tracing, since the profiler may still be reading it.
  // TODO(benvanik): move to an atexit handler instead, verify with ASAN/MSAN
  // TODO(scotttodd): Make this compatible with testing:
  //     two test cases, one for each function in the same executable
  //     first test case passes, second fails to open the file (already open)
#else
  delete executable->library;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  for (iree_host_size_t i = 0; i < executable->temp_file_count; ++i) {
    iree_string_view_t file_path = executable->temp_files[i];
    iree::file_io::DeleteFile(std::string(file_path.data, file_path.size))
        .IgnoreError();
    iree_allocator_free(host_allocator, (void*)file_path.data);
  }

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_legacy_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_local_executable_call_t* call) {
  iree_hal_legacy_executable_t* executable =
      (iree_hal_legacy_executable_t*)base_executable;

  if (IREE_UNLIKELY(ordinal >= executable->entry_fn_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  flatbuffers_string_t entry_point_str = flatbuffers_string_vec_at(
      iree_DyLibExecutableDef_entry_points_get(executable->def), ordinal);
  iree_string_view_t entry_point_name = iree_make_string_view(
      entry_point_str, flatbuffers_string_len(entry_point_str));
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_dylib_call");
  }
  IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, entry_point_name.data,
                                      entry_point_name.size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  executable->entry_fns[ordinal](call->bindings, call->push_constants,
                                 (const uint32_t*)&call->workgroup_id,
                                 (const uint32_t*)&call->workgroup_count,
                                 (const uint32_t*)&call->workgroup_size);

  IREE_TRACE_ZONE_END(z0);

  return iree_ok_status();
}

const iree_hal_local_executable_vtable_t iree_hal_legacy_executable_vtable = {
    /*.base=*/
    {
        /*.destroy=*/iree_hal_legacy_executable_destroy,
    },
    /*.issue_call=*/iree_hal_legacy_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_legacy_library_loader_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_legacy_library_loader_t;

extern const iree_hal_executable_loader_vtable_t
    iree_hal_legacy_library_loader_vtable;

iree_status_t iree_hal_legacy_library_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_legacy_library_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_legacy_library_loader_vtable, &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_legacy_library_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_legacy_library_loader_t* executable_loader =
      (iree_hal_legacy_library_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_legacy_library_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_hal_executable_format_t executable_format) {
  return executable_format == iree_hal_make_executable_format("DLIB");
}

static iree_status_t iree_hal_legacy_library_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_legacy_library_loader_t* executable_loader =
      (iree_hal_legacy_library_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify and fetch the executable flatbuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_executable_flatbuffer_verify(
              executable_spec->executable_data,
              executable_spec->executable_layout_count));
  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(executable_spec->executable_data.data);

  // Perform the load (and requisite disgusting hackery).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_legacy_executable_create(
              executable_def, executable_spec->executable_layout_count,
              executable_spec->executable_layouts,
              executable_loader->host_allocator, out_executable));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

const iree_hal_executable_loader_vtable_t
    iree_hal_legacy_library_loader_vtable = {
        /*.destroy=*/iree_hal_legacy_library_loader_destroy,
        /*.query_support=*/iree_hal_legacy_library_loader_query_support,
        /*.try_load=*/iree_hal_legacy_library_loader_try_load,
};
