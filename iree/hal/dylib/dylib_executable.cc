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

#include "flatbuffers/flatbuffers.h"
#include "iree/base/file_io.h"
#include "iree/base/file_path.h"
#include "iree/schemas/dylib_executable_def_generated.h"

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

  auto dylib_executable_def =
      ::flatbuffers::GetRoot<DyLibExecutableDef>(spec.executable_data.data());

  if (!dylib_executable_def->entry_points() ||
      dylib_executable_def->entry_points()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No entry points defined";
  }
  if (!dylib_executable_def->library_embedded() ||
      dylib_executable_def->library_embedded()->size() == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No embedded library";
  }

  // Write the embedded library out to a temp file, since all of the dynamic
  // library APIs work with files. We could instead use in-memory files on
  // platforms where that is convenient.
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

  absl::string_view embedded_library_data(
      reinterpret_cast<const char*>(
          dylib_executable_def->library_embedded()->data()),
      dylib_executable_def->library_embedded()->size());
  IREE_RETURN_IF_ERROR(
      file_io::SetFileContents(library_temp_path, embedded_library_data));

  IREE_ASSIGN_OR_RETURN(executable_library_,
                        DynamicLibrary::Load(library_temp_path.c_str()));

  if (dylib_executable_def->debug_database_filename() &&
      dylib_executable_def->debug_database_embedded()) {
    IREE_TRACE_SCOPE0("DyLibExecutable::AttachDebugDatabase");
    absl::string_view debug_database_filename(
        dylib_executable_def->debug_database_filename()->data(),
        dylib_executable_def->debug_database_filename()->size());
    absl::string_view debug_database_data(
        reinterpret_cast<const char*>(
            dylib_executable_def->debug_database_embedded()->data()),
        dylib_executable_def->debug_database_embedded()->size());
    auto debug_database_path = file_path::JoinPaths(
        file_path::DirectoryName(library_temp_path), debug_database_filename);
    temp_file_paths_.push_back(debug_database_path);
    IREE_IGNORE_ERROR(
        file_io::SetFileContents(debug_database_path, debug_database_data));
    executable_library_->AttachDebugDatabase(debug_database_path.c_str());
  }

  const auto& entry_points = *dylib_executable_def->entry_points();
  entry_functions_.resize(entry_points.size());
  IREE_TRACE(entry_names_.resize(entry_points.size()));
  for (int i = 0; i < entry_functions_.size(); ++i) {
    void* symbol = executable_library_->GetSymbol(entry_points[i]->c_str());
    if (!symbol) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Could not find symbol: " << entry_points[i];
    }
    entry_functions_[i] = symbol;

    IREE_TRACE(entry_names_[i] = entry_points[i]->c_str());
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
