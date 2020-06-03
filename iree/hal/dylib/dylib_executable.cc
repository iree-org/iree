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
StatusOr<ref_ptr<DyLibExecutable>> DyLibExecutable::Load(
    hal::Allocator* allocator, ExecutableSpec spec, bool allow_aliasing_data) {
  auto executable = make_ref<DyLibExecutable>(spec, allow_aliasing_data);
  RETURN_IF_ERROR(executable->Initialize());
  return executable;
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
  ASSIGN_OR_RETURN(std::string temp_file, file_io::GetTempFile(base_name));
  // Add platform-specific file extensions so opinionated dynamic library
  // loaders are more likely to find the file:
#if defined(IREE_PLATFORM_WINDOWS)
  temp_file += ".dll";
#else
  temp_file += ".so";
#endif

  absl::string_view embedded_library_data(
      reinterpret_cast<const char*>(
          dylib_executable_def->library_embedded()->data()),
      dylib_executable_def->library_embedded()->size());
  RETURN_IF_ERROR(file_io::SetFileContents(temp_file, embedded_library_data));

  ASSIGN_OR_RETURN(executable_library_,
                   DynamicLibrary::Load(temp_file.c_str()));

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

DyLibExecutable::DyLibExecutable(ExecutableSpec spec, bool allow_aliasing_data)
    : spec_(spec) {
  if (!allow_aliasing_data) {
    // Clone data.
    cloned_executable_data_ = {spec.executable_data.begin(),
                               spec.executable_data.end()};
    spec_.executable_data = absl::MakeConstSpan(cloned_executable_data_);
  }
}

// TODO(scotttodd): delete temp file after unloading library?
DyLibExecutable::~DyLibExecutable() = default;

Status DyLibExecutable::Invoke(int func_id, absl::Span<void*> args) const {
  return UnimplementedErrorBuilder(IREE_LOC) << "DyLibExecutable::Invoke NYI";
}

}  // namespace dylib
}  // namespace hal
}  // namespace iree
