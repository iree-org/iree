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
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace cuda {

static const char* kCudaLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nvcuda.dll",
#else
    "libcuda.so",
#endif
};

Status DynamicSymbols::LoadSymbols() {
  IREE_TRACE_SCOPE();

  IREE_RETURN_IF_ERROR(DynamicLibrary::Load(
      absl::MakeSpan(kCudaLoaderSearchNames), &loader_library_));

#define CU_PFN_DECL(cudaSymbolName)                                         \
  {                                                                         \
    using FuncPtrT = std::add_pointer<decltype(::cudaSymbolName)>::type;    \
    static const char* kName = #cudaSymbolName;                             \
    cudaSymbolName = loader_library_->GetSymbol<FuncPtrT>(kName);           \
    if (!cudaSymbolName) {                                                  \
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "symbol not found"); \
    }                                                                       \
  }

#include "dynamic_symbols_tables.h"
#undef CU_PFN_DECL

  return OkStatus();
}

}  // namespace cuda
}  // namespace hal
}  // namespace iree
