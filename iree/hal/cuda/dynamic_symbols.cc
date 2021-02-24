// Copyright 2021 Google LLC
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

#include "iree/hal/cuda/dynamic_symbols.h"

#include <cstddef>

#include "absl/types/span.h"
#include "iree/base/dynamic_library.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static const char* kCUDALoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nvcuda.dll",
#else
    "libcuda.so",
#endif
};

extern "C" {

iree_status_t load_symbols(iree_hal_cuda_dynamic_symbols_t* syms) {
  std::unique_ptr<iree::DynamicLibrary> loader_library;
  IREE_RETURN_IF_ERROR(iree::DynamicLibrary::Load(
      absl::MakeSpan(kCUDALoaderSearchNames), &loader_library));

#define CU_PFN_DECL(cudaSymbolName, ...)                                    \
  {                                                                         \
    using FuncPtrT = decltype(syms->cudaSymbolName);                        \
    static const char* kName = #cudaSymbolName;                             \
    syms->cudaSymbolName = loader_library->GetSymbol<FuncPtrT>(kName);      \
    if (!syms->cudaSymbolName) {                                            \
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "symbol not found"); \
    }                                                                       \
  }

#include "dynamic_symbols_tables.h"
#undef CU_PFN_DECL
  syms->opaque_loader_library_ = (void*)loader_library.release();
  return iree_ok_status();
}

void unload_symbols(iree_hal_cuda_dynamic_symbols_t* syms) {
  delete (iree::DynamicLibrary*)syms->opaque_loader_library_;
}

}  // extern "C"