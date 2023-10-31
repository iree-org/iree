// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include <cstring>

#include "iree/base/internal/path.h"  // TODO: Make public trampoline.
#include "iree_pjrt/common/api_impl.h"
//#include "iree/vm/api.h"
//#include "iree_pjrt/common/pjrt_plugin_defs.h"
//#include "iree_pjrt/common/pjrt_plugin_impl.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#if (defined(_WIN32) || defined(__CYGWIN__)) && \
    !defined(PJRT_PLUGIN_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define PJRT_PLUGIN_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if PJRT_PLUGIN_BUILDING_LIBRARY
#define PJRT_PLUGIN_EXPORTED __declspec(dllexport)
#else
#define PJRT_PLUGIN_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#if PJRT_PLUGIN_BUILDING_LIBRARY
#define PJRT_PLUGIN_EXPORTED __attribute__((visibility("default")))
#else
#define PJRT_PLUGIN_EXPORTED
#endif
#endif

//===----------------------------------------------------------------------===//
// Entry-point private API.
//===----------------------------------------------------------------------===//

namespace iree::pjrt {
namespace {

// The including module must define `InitializeAPI` to set up the
// implementation.
void InitializeAPI(PJRT_Api* api);

// Gets the dylib-scope static API struct.
PJRT_Api* GetRawAPIStruct() {
  static PJRT_Api static_api;
  return &static_api;
}

// Gets the dylib-scope initialized API struct.
PJRT_Api* GetInitializedAPIStruct() {
  static PJRT_Api* inited_api = ([]() {
    PJRT_Api* api = GetRawAPIStruct();
    memset(api, 0, sizeof(PJRT_Api));
    InitializeAPI(api);
    return api;
  })();
  return inited_api;
}

}  // namespace
}  // namespace iree::pjrt

//===----------------------------------------------------------------------===//
// Export function typedefs.
// TODO: These should probably be included in the PJRT C API.
//===----------------------------------------------------------------------===//

// Gets the version of the plugin API. This is incremented on breaking changes
// to the exported plugin API.
// Exported as: PJRT_Plugin_ApiVersion
typedef unsigned PJRT_Plugin_ApiVersion_FN();

// Primary entry point for creating a specific PJRT API instance.
// Exported as: PJRT_Plugin_Create
typedef PJRT_Api* PJRT_Plugin_Create_FN();

//===----------------------------------------------------------------------===//
// Declarations (needed to set visibility and import/export flags).
//===----------------------------------------------------------------------===//

extern "C" {
PJRT_PLUGIN_EXPORTED unsigned GetPjrtApiVersion();
static_assert(std::is_same_v<decltype(GetPjrtApiVersion),
                             PJRT_Plugin_ApiVersion_FN>);

PJRT_PLUGIN_EXPORTED PJRT_Api* GetPjrtApi();
static_assert(
    std::is_same_v<decltype(GetPjrtApi), PJRT_Plugin_Create_FN>);
}

//===----------------------------------------------------------------------===//
// Exports from the shared library
//===----------------------------------------------------------------------===//

unsigned GetPjrtApiVersion() { return 1; }

PJRT_Api* GetPjrtApi() {
  return iree::pjrt::GetInitializedAPIStruct();
}
