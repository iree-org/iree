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

#include "iree/hal/dylib/dylib_executable.h"

#include "iree/base/file_io.h"
#include "iree/base/file_path.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/dylib_executable_def_reader.h"
#include "iree/schemas/dylib_executable_def_verifier.h"

// NOTE: starting to port this to C.

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_dylib_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
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

namespace iree {
namespace hal {
namespace dylib {

// static
StatusOr<ref_ptr<DyLibExecutable>> DyLibExecutable::Load(ExecutableSpec spec) {
  auto executable = make_ref<DyLibExecutable>();
  IREE_RETURN_IF_ERROR(executable->Initialize(spec));
  return executable;
}

DyLibExecutable::DyLibExecutable() = default;

DyLibExecutable::~DyLibExecutable() {
  IREE_TRACE_SCOPE0("DyLibExecutable::dtor");
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  // Leak the library when tracing, since the profiler may still be reading it.
  // TODO(benvanik): move to an atexit handler instead, verify with ASAN/MSAN
  // TODO(scotttodd): Make this compatible with testing:
  //     two test cases, one for each function in the same executable
  //     first test case passes, second fails to open the file (already open)
  executable_library_.release();
#else
  executable_library_.reset();
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  for (const auto& file_path : temp_file_paths_) {
    file_io::DeleteFile(file_path).IgnoreError();
  }
}

Status DyLibExecutable::Initialize(ExecutableSpec spec) {
  IREE_TRACE_SCOPE0("DyLibExecutable::Initialize");

  // Verify and fetch the executable flatbuffer wrapper.
  iree_const_byte_span_t executable_data = iree_make_const_byte_span(
      spec.executable_data.data(), spec.executable_data.size());
  IREE_RETURN_IF_ERROR(
      iree_hal_dylib_executable_flatbuffer_verify(executable_data));
  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(executable_data.data);

  // Write the embedded library out to a temp file, since all of the dynamic
  // library APIs work with files. We could instead use in-memory files on
  // platforms where that is convenient.
  //
  // TODO(#3845): use dlopen on an fd with either dlopen(/proc/self/fd/NN),
  // fdlopen, or android_dlopen_ext to avoid needing to write the file to disk.
  // Can fallback to memfd_create + dlopen where available, and fallback from
  // that to disk (maybe just windows/mac).
  std::string base_name = "dylib_executable";
  IREE_ASSIGN_OR_RETURN(auto library_temp_path,
                        file_io::GetTempFile(base_name));
  temp_file_paths_.push_back(library_temp_path);

// Add platform-specific file extensions so opinionated dynamic library
// loaders are more likely to find the file:
#if defined(IREE_PLATFORM_WINDOWS)
  library_temp_path += ".dll";
#else
  library_temp_path += ".so";
#endif

  flatbuffers_uint8_vec_t embedded_library_vec =
      iree_DyLibExecutableDef_library_embedded_get(executable_def);
  IREE_RETURN_IF_ERROR(file_io::SetFileContents(
      library_temp_path,
      absl::string_view(reinterpret_cast<const char*>(embedded_library_vec),
                        flatbuffers_uint8_vec_len(embedded_library_vec))));

  IREE_ASSIGN_OR_RETURN(executable_library_,
                        DynamicLibrary::Load(library_temp_path.c_str()));

  flatbuffers_string_t debug_database_filename =
      iree_DyLibExecutableDef_debug_database_filename_get(executable_def);
  flatbuffers_uint8_vec_t debug_database_embedded_vec =
      iree_DyLibExecutableDef_debug_database_embedded_get(executable_def);
  if (flatbuffers_string_len(debug_database_filename) &&
      flatbuffers_uint8_vec_len(debug_database_embedded_vec)) {
    IREE_TRACE_SCOPE0("DyLibExecutable::AttachDebugDatabase");
    auto debug_database_path = file_path::JoinPaths(
        file_path::DirectoryName(library_temp_path),
        absl::string_view(debug_database_filename,
                          flatbuffers_string_len(debug_database_filename)));
    temp_file_paths_.push_back(debug_database_path);
    IREE_IGNORE_ERROR(file_io::SetFileContents(
        debug_database_path,
        absl::string_view(
            reinterpret_cast<const char*>(debug_database_embedded_vec),
            flatbuffers_uint8_vec_len(debug_database_embedded_vec))));
    executable_library_->AttachDebugDatabase(debug_database_path.c_str());
  }

  flatbuffers_string_vec_t entry_points =
      iree_DyLibExecutableDef_entry_points_get(executable_def);
  entry_functions_.resize(flatbuffers_string_vec_len(entry_points));
  IREE_TRACE(entry_names_.resize(flatbuffers_string_vec_len(entry_points)));
  for (size_t i = 0; i < entry_functions_.size(); ++i) {
    flatbuffers_string_t entry_point =
        flatbuffers_string_vec_at(entry_points, i);
    void* symbol = executable_library_->GetSymbol(entry_point);
    if (!symbol) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Could not find symbol: " << entry_point;
    }
    entry_functions_[i] = symbol;

    IREE_TRACE(entry_names_[i] = entry_point);
  }

  return OkStatus();
}

struct DyLibDispatchState : public HostExecutable::DispatchState {
  DyLibDispatchState() = default;

  IREE_TRACE(const char* entry_name = nullptr);

  void* entry_function = nullptr;
  std::array<void*, 32> args;
  std::array<uint32_t, 32> push_constants;
};

StatusOr<ref_ptr<HostExecutable::DispatchState>>
DyLibExecutable::PrepareDispatch(const DispatchParams& params) {
  IREE_TRACE_SCOPE0("DyLibExecutable::PrepareDispatch");

  if (params.entry_point >= entry_functions_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Invalid entry point ordinal " << params.entry_point;
  }

  auto dispatch_state = make_ref<DyLibDispatchState>();
  IREE_TRACE(dispatch_state->entry_name = entry_names_[params.entry_point]);
  dispatch_state->entry_function = entry_functions_[params.entry_point];

  int binding_count = 0;
  for (size_t set = 0; set < params.set_bindings.size(); ++set) {
    for (size_t binding = 0; binding < params.set_bindings[set].size();
         ++binding) {
      const auto& io_binding = params.set_bindings[set][binding];
      IREE_ASSIGN_OR_RETURN(auto memory,
                            io_binding.buffer->MapMemory<uint8_t>(
                                MemoryAccessBitfield::kWrite, io_binding.offset,
                                io_binding.length));
      auto data = memory.mutable_data();
      dispatch_state->args[binding_count++] = data;
    }
  }
  dispatch_state->push_constants = params.push_constants->values;

  return std::move(dispatch_state);
}

Status DyLibExecutable::DispatchTile(DispatchState* state,
                                     std::array<uint32_t, 3> workgroup_xyz) {
  auto* dispatch_state = static_cast<DyLibDispatchState*>(state);
  IREE_TRACE_SCOPE_DYNAMIC(dispatch_state->entry_name);

  auto entry_function = (void (*)(void**, uint32_t*, int32_t, int32_t,
                                  int32_t))dispatch_state->entry_function;
  entry_function(dispatch_state->args.data(),
                 dispatch_state->push_constants.data(), workgroup_xyz[0],
                 workgroup_xyz[1], workgroup_xyz[2]);

  return OkStatus();
}

}  // namespace dylib
}  // namespace hal
}  // namespace iree
