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

#include "absl/memory/memory.h"
#include "iree/base/dynamic_library.h"
#include "iree/base/platform_headers.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_WINDOWS)

namespace iree {

class DynamicLibraryWin : public DynamicLibrary {
 public:
  ~DynamicLibraryWin() override {
    // TODO(benvanik): disable if we want to get profiling results.
    //   Sometimes closing the library can prevent proper symbolization on
    //   crashes or in sampling profilers.
    ::FreeLibrary(library_);
  }

  static StatusOr<std::unique_ptr<DynamicLibrary>> TryLoad(
      const char* file_name) {
    HMODULE library = ::LoadLibraryA(file_name);
    if (library) {
      return absl::WrapUnique(new DynamicLibraryWin(file_name, library));
    }

    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to open dynamic library, not found on search path";
  }

  void* GetSymbol(const char* symbol_name) const override {
    return reinterpret_cast<void*>(::GetProcAddress(library_, symbol_name));
  }

 private:
  DynamicLibraryWin(std::string file_name, HMODULE library)
      : DynamicLibrary(file_name), library_(library) {}

  HMODULE library_;
};

// static
StatusOr<std::unique_ptr<DynamicLibrary>> DynamicLibrary::TryLoad(
    const char* file_name) {
  return DynamicLibraryWin::TryLoad(file_name);
}

}  // namespace iree

#endif  // IREE_PLATFORM_*
