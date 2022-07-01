// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/dynamic_symbols.h"

#include <string.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static const char* kLevelZeroLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    NULL,
#else
    "libze_loader.so",
#endif
};

static iree_status_t iree_hal_level_zero_dynamic_symbols_resolve_all(
    iree_hal_level_zero_dynamic_symbols_t* syms) {
#define ZE_PFN_DECL(levelZeroSymbolName, ...)                              \
  {                                                                        \
    static const char* kName = #levelZeroSymbolName;                       \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(               \
        syms->loader_library, kName, (void**)&syms->levelZeroSymbolName)); \
  }
#include "experimental/level_zero/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef ZE_PFN_DECL
  return iree_ok_status();
}

iree_status_t iree_hal_level_zero_dynamic_symbols_initialize(
    iree_allocator_t allocator,
    iree_hal_level_zero_dynamic_symbols_t* out_syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kLevelZeroLoaderSearchNames), kLevelZeroLoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, allocator, &out_syms->loader_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "LevelZero runtime library not available; ensure "
                            "installed and on path");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_level_zero_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_level_zero_dynamic_symbols_deinitialize(out_syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_level_zero_dynamic_symbols_deinitialize(
    iree_hal_level_zero_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_dynamic_library_release(syms->loader_library);
  memset(syms, 0, sizeof(*syms));
  IREE_TRACE_ZONE_END(z0);
}
