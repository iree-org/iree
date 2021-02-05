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
#include "absl/strings/str_replace.h"
#include "iree/base/dynamic_library.h"
#include "iree/base/internal/file_path.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_WINDOWS)

// TODO(benvanik): support PDB overlays when tracy is not enabled too; we'll
// need to rearrange how the dbghelp lock is handled for that (probably moving
// it here and having the tracy code redirect to this).
#if defined(TRACY_ENABLE)
#define IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT 1
#pragma warning(disable : 4091)
#include <dbghelp.h>
extern "C" void IREEDbgHelpLock();
extern "C" void IREEDbgHelpUnlock();
#endif  // TRACY_ENABLE

namespace iree {

// We need to match the expected paths from dbghelp exactly or else we'll get
// spurious warnings during module resolution. AFAICT our approach here with
// loading the PDBs directly with a module base/size of the loaded module will
// work regardless but calls like SymRefreshModuleList will still attempt to
// load from system symbol search paths if things don't line up.
static void CanonicalizePath(std::string* path) {
  absl::StrReplaceAll({{"/", "\\"}}, path);
  absl::StrReplaceAll({{"\\\\", "\\"}}, path);
}

class DynamicLibraryWin : public DynamicLibrary {
 public:
  ~DynamicLibraryWin() override {
    IREE_TRACE_SCOPE();
    // TODO(benvanik): disable if we want to get profiling results.
    //   Sometimes closing the library can prevent proper symbolization on
    //   crashes or in sampling profilers.
    ::FreeLibrary(library_);
  }

  static Status Load(absl::Span<const char* const> search_file_names,
                     std::unique_ptr<DynamicLibrary>* out_library) {
    IREE_TRACE_SCOPE();
    out_library->reset();

    for (int i = 0; i < search_file_names.size(); ++i) {
      HMODULE library = ::LoadLibraryA(search_file_names[i]);
      if (library) {
        out_library->reset(
            new DynamicLibraryWin(search_file_names[i], library));
        return OkStatus();
      }
    }

    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "unable to open dynamic library, not found on search paths");
  }

#if defined(IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT)
  void AttachDebugDatabase(const char* database_file_name) override {
    IREE_TRACE_SCOPE();

    // Derive the base module name (path stem) for the loaded module.
    // For example, the name of 'C:\Dev\foo.dll' would be 'foo'.
    // This name is used by dbghelp for listing loaded modules and we want to
    // ensure we match the name of the PDB module with the library module.
    std::string module_name = file_name_;
    size_t last_slash = module_name.find_last_of('\\');
    if (last_slash != std::string::npos) {
      module_name = module_name.substr(last_slash + 1);
    }
    size_t dot = module_name.find_last_of('.');
    if (dot != std::string::npos) {
      module_name = module_name.substr(0, dot);
    }

    IREEDbgHelpLock();

    // Useful for debugging; will print search paths and results:
    // SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEBUG);

    // Enumerates all loaded modules in the process to extract the module
    // base/size parameters we need to overlay the PDB. There's other ways to
    // get this (such as registering a LdrDllNotification callback and snooping
    // the values during LoadLibrary or using CreateToolhelp32Snapshot), however
    // EnumerateLoadedModules is in dbghelp which we are using anyway.
    ModuleEnumCallbackState state;
    state.module_file_path = file_name_.c_str();
    EnumerateLoadedModules64(GetCurrentProcess(), EnumLoadedModulesCallback,
                             &state);

    // Load the PDB file and overlay it onto the already-loaded module at the
    // address range it got loaded into.
    if (state.module_base != 0) {
      SymLoadModuleEx(GetCurrentProcess(), NULL, database_file_name,
                      module_name.c_str(), state.module_base, state.module_size,
                      NULL, 0);
    }

    IREEDbgHelpUnlock();
  }
#endif  // IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT

  void* GetSymbol(const char* symbol_name) const override {
    return reinterpret_cast<void*>(::GetProcAddress(library_, symbol_name));
  }

 private:
  DynamicLibraryWin(std::string file_name, HMODULE library)
      : DynamicLibrary(std::move(file_name)), library_(library) {
    CanonicalizePath(&file_name_);
  }

#if defined(IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT)
  struct ModuleEnumCallbackState {
    const char* module_file_path = NULL;
    DWORD64 module_base = 0;
    ULONG module_size = 0;
  };
  static BOOL EnumLoadedModulesCallback(PCSTR ModuleName, DWORD64 ModuleBase,
                                        ULONG ModuleSize, PVOID UserContext) {
    auto* state = reinterpret_cast<ModuleEnumCallbackState*>(UserContext);
    if (strcmp(ModuleName, state->module_file_path) != 0) {
      return TRUE;  // not a match; continue
    }
    state->module_base = ModuleBase;
    state->module_size = ModuleSize;
    return FALSE;  // match found; stop enumeration
  }
#endif  // IREE_HAVE_DYNAMIC_LIBRARY_PDB_SUPPORT

  HMODULE library_ = NULL;
};

// static
Status DynamicLibrary::Load(absl::Span<const char* const> search_file_names,
                            std::unique_ptr<DynamicLibrary>* out_library) {
  return DynamicLibraryWin::Load(search_file_names, out_library);
}

}  // namespace iree

#endif  // IREE_PLATFORM_*
