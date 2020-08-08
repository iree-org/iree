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

#ifndef IREE_BASE_DYNAMIC_LIBRARY_H_
#define IREE_BASE_DYNAMIC_LIBRARY_H_

#include <memory>
#include <string>

#include "absl/types/span.h"
#include "iree/base/status.h"

namespace iree {

// Dynamic library / shared object cross-platform wrapper class.
//
// Paths searched for libraries are platform and environment-specific.
// In general...
//   * On Linux, the LD_LIBRARY_PATH environment variable may be set to a
//     colon-separated list of directories to search before the standard set.
//   * On Windows, all directories in the PATH environment variable are checked.
// Library file names may be relative to the search paths, or absolute.
// Certain platforms may require the library extension (.so, .dll), or it may
// be optional. If you know the extension, prefer to include it.
//
// Usage:
//   static const char* kSearchNames[] = {"libfoo.so"};
//   IREE_ASSIGN_OR_RETURN(library,
//                         DynamicLibrary::Load(absl::MakeSpan(kSearchNames)));
//   void* library_symbol_bar = library->GetSymbol("bar");
//   void* library_symbol_baz = library->GetSymbol("baz");
class DynamicLibrary {
 public:
  virtual ~DynamicLibrary() = default;

  // Loads the library at the null-terminated string |search_file_name|.
  static StatusOr<std::unique_ptr<DynamicLibrary>> Load(
      const char* search_file_name) {
    return Load(absl::Span<const char* const>({search_file_name}));
  }
  // Loads the library at the first name within |search_file_names| found.
  static StatusOr<std::unique_ptr<DynamicLibrary>> Load(
      absl::Span<const char* const> search_file_names);

  // Gets the name of the library file that is loaded.
  const std::string& file_name() const { return file_name_; }

  // Gets the address of a symbol with the given name in the loaded library.
  // Returns NULL if the symbol could not be found.
  virtual void* GetSymbol(const char* symbol_name) const = 0;
  template <typename T>
  T GetSymbol(const char* symbol_name) const {
    return reinterpret_cast<T>(GetSymbol(symbol_name));
  }

 protected:
  // Private constructor, use |Load| factory method instead.
  DynamicLibrary(std::string file_name) : file_name_(file_name) {}

  std::string file_name_;
};

}  // namespace iree

#endif  // IREE_BASE_DYNAMIC_LIBRARY_H_
