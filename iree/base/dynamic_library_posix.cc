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
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)

#include <dlfcn.h>

namespace iree {

class DynamicLibraryPosix : public DynamicLibrary {
 public:
  ~DynamicLibraryPosix() override {
    // TODO(benvanik): disable if we want to get profiling results.
    //   Sometimes closing the library can prevent proper symbolization on
    //   crashes or in sampling profilers.
    ::dlclose(library_);
  }

  static Status Load(absl::Span<const char* const> search_file_names,
                     std::unique_ptr<DynamicLibrary>* out_library) {
    IREE_TRACE_SCOPE0("DynamicLibraryPosix::Load");
    out_library->reset();

    for (int i = 0; i < search_file_names.size(); ++i) {
      void* library = ::dlopen(search_file_names[i], RTLD_LAZY | RTLD_LOCAL);
      if (library) {
        out_library->reset(
            new DynamicLibraryPosix(search_file_names[i], library));
        return OkStatus();
      }
    }
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "unable to open dynamic library:'%s'", dlerror());
  }

  void* GetSymbol(const char* symbol_name) const override {
    return ::dlsym(library_, symbol_name);
  }

 private:
  DynamicLibraryPosix(std::string file_name, void* library)
      : DynamicLibrary(file_name), library_(library) {}

  void* library_;
};

// static
Status DynamicLibrary::Load(absl::Span<const char* const> search_file_names,
                            std::unique_ptr<DynamicLibrary>* out_library) {
  return DynamicLibraryPosix::Load(search_file_names, out_library);
}

}  // namespace iree

#endif  // IREE_PLATFORM_*
