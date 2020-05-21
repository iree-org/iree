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

#include "iree/base/dynamic_library.h"

#include "iree/base/tracing.h"

namespace iree {

// static
StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::Load(
    const char* file_name) {
  return Load(absl::Span<const char* const>({file_name}));
}

// static
StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::Load(
    const std::string& file_name) {
  return Load(file_name.c_str());
}

// static
StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::Load(
    absl::Span<const char* const> file_names) {
  IREE_TRACE_SCOPE0("DynamicLibrary::Load");

  for (int i = 0; i < file_names.size(); ++i) {
    auto library_or = TryLoad(file_names[i]);
    if (library_or.ok()) {
      return std::move(library_or.value());
    }
  }

  return UnavailableErrorBuilder(IREE_LOC)
         << "Unable to open dynamic library, not found on search paths";
}

// static
StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::Load(
    const std::vector<std::string>& file_names) {
  std::vector<const char*> file_names_cstrs(file_names.size());
  for (int i = 0; i < file_names.size(); ++i) {
    file_names_cstrs[i] = file_names[i].c_str();
  }
  return Load(file_names_cstrs);
}

}  // namespace iree
