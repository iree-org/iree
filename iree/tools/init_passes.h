// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines a helper to add passes to the global registry.

#ifndef IREE_TOOLS_INIT_PASSES_H_
#define IREE_TOOLS_INIT_PASSES_H_

#include <cstdlib>

#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Dialect/HAL/Conversion/Passes.h"
#include "iree/tools/init_iree_passes.h"
#include "iree/tools/init_mlir_passes.h"

namespace mlir {
namespace iree_compiler {

// Registers IREE core passes and other important passes to the global registry.
inline void registerAllPasses() {
  registerAllIreePasses();
  registerConversionPasses();
  registerMlirPasses();
  registerHALConversionPasses();
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_PASSES_H_
