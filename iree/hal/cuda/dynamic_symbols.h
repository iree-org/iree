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

#ifndef IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "iree/base/dynamic_library.h"
#include "iree/base/status.h"
#include "iree/hal/cuda/cuda_headers.h"

namespace iree {
namespace hal {
namespace cuda {

/// DyanmicSymbols allow loading dynamically a subset of CUDA driver API. It
/// loads all the function declared in `dynamic_symbol_tables.def` and fail if
/// any of the symbol is not available. The functions signatures are matching
/// the declarations in `cuda.h`.
struct DynamicSymbols {
  Status LoadSymbols();

#define CU_PFN_DECL(cudaSymbolName) \
  std::add_pointer<decltype(::cudaSymbolName)>::type cudaSymbolName;

#include "dynamic_symbols_tables.h"
#undef CU_PFN_DECL

 private:
  // Cuda Loader dynamic library.
  std::unique_ptr<DynamicLibrary> loader_library_;
};

}  // namespace cuda
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_
