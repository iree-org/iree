// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/version.h"

#include <string_view>

std::string mlir::iree_compiler::getIreeRevision() {
#ifdef IREE_RELEASE_VERSION
#ifdef IREE_RELEASE_REVISION
  if constexpr (std::string_view(IREE_RELEASE_REVISION) == "HEAD") {
    return IREE_RELEASE_VERSION;
  }
  return IREE_RELEASE_VERSION " @ " IREE_RELEASE_REVISION;
#else
  return IREE_RELEASE_VERSION;
#endif
#else
  return "";
#endif
}
