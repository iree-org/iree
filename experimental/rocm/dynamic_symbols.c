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

#include "experimental/rocm/dynamic_symbols.h"

#include <stddef.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static const char *kROCMLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "amdhip64.dll",
#else
    "libamdhip64.so",
#endif
};

static iree_status_t iree_hal_rocm_dynamic_symbols_resolve_all(
    iree_hal_rocm_dynamic_symbols_t *syms) {
#define RC_PFN_DECL(rocmSymbolName, ...)                               \
  {                                                                    \
    static const char *kName = #rocmSymbolName;                        \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(           \
        syms->loader_library, kName, (void **)&syms->rocmSymbolName)); \
  }
#define RC_PFN_STR_DECL(rocmSymbolName, ...) RC_PFN_DECL(rocmSymbolName, ...)
#include "experimental/rocm/dynamic_symbol_tables.h"
#undef RC_PFN_DECL
#undef RC_PFN_STR_DECL
  return iree_ok_status();
}

iree_status_t iree_hal_rocm_dynamic_symbols_initialize(
    iree_allocator_t allocator, iree_hal_rocm_dynamic_symbols_t *out_syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kROCMLoaderSearchNames), kROCMLoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, allocator, &out_syms->loader_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "ROCM/HIP runtime library not available; ensure installed and on path");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_rocm_dynamic_symbols_deinitialize(out_syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_rocm_dynamic_symbols_deinitialize(
    iree_hal_rocm_dynamic_symbols_t *syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_dynamic_library_release(syms->loader_library);
  memset(syms, 0, sizeof(*syms));
  IREE_TRACE_ZONE_END(z0);
}
