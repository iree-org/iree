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

#include "iree/base/api.h"
#include "iree/hal/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
/// DyanmicSymbols allow loading dynamically a subset of CUDA driver API. It
/// loads all the function declared in `dynamic_symbol_tables.def` and fail if
/// any of the symbol is not available. The functions signatures are matching
/// the declarations in `cuda.h`.
typedef struct {
#define CU_PFN_DECL(cudaSymbolName, ...) \
  CUresult (*cudaSymbolName)(__VA_ARGS__);
#include "dynamic_symbols_tables.h"
#undef CU_PFN_DECL
  void* opaque_loader_library_;
} iree_hal_cuda_dynamic_symbols_t;

iree_status_t load_symbols(iree_hal_cuda_dynamic_symbols_t* syms);
void unload_symbols(iree_hal_cuda_dynamic_symbols_t* syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_DYNAMIC_SYMBOLS_H_
