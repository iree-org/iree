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
#include "iree/schemas/dylib_executable_def_generated.h"

namespace iree {
namespace hal {
namespace dylib {

// static
StatusOr<ref_ptr<DyLibExecutable>> DyLibExecutable::Load(ExecutableSpec spec) {
  auto executable = make_ref<DyLibExecutable>(spec);
  RETURN_IF_ERROR(executable->Initialize());
  return executable;
}

DyLibExecutable::DyLibExecutable(ExecutableSpec spec) : spec_(spec) {}

DyLibExecutable::~DyLibExecutable() {
  executable_library_.reset();
  if (!executable_library_temp_path_.empty()) {
    file_io::DeleteFile(executable_library_temp_path_).IgnoreError();
  }
}

Status DyLibExecutable::Initialize() {
  auto dylib_executable_def =
      ::flatbuffers::GetRoot<DyLibExecutableDef>(spec_.executable_data.data());

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
  ASSIGN_OR_RETURN(executable_library_temp_path_,
                   file_io::GetTempFile(base_name));
  // Add platform-specific file extensions so opinionated dynamic library
  // loaders are more likely to find the file:
#if defined(IREE_PLATFORM_WINDOWS)
  executable_library_temp_path_ += ".dll";
#else
  executable_library_temp_path_ += ".so";
#endif

  absl::string_view embedded_library_data(
      reinterpret_cast<const char*>(
          dylib_executable_def->library_embedded()->data()),
      dylib_executable_def->library_embedded()->size());
  RETURN_IF_ERROR(file_io::SetFileContents(executable_library_temp_path_,
                                           embedded_library_data));

  ASSIGN_OR_RETURN(executable_library_,
                   DynamicLibrary::Load(executable_library_temp_path_.c_str()));

  const auto& entry_points = *dylib_executable_def->entry_points();
  entry_functions_.resize(entry_points.size());
  for (int i = 0; i < entry_functions_.size(); ++i) {
    void* symbol = executable_library_->GetSymbol(entry_points[i]->c_str());
    if (!symbol) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Could not find symbol: " << entry_points[i];
    }
    entry_functions_[i] = symbol;
  }

  return OkStatus();
}

Status DyLibExecutable::Invoke(int func_id, absl::Span<void*> args) const {
  return UnimplementedErrorBuilder(IREE_LOC) << "DyLibExecutable::Invoke NYI";
}

}  // namespace dylib
}  // namespace hal
}  // namespace iree
