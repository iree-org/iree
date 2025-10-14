// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

namespace IREE::PCF {
namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"
} // namespace
} // namespace IREE::PCF

void registerPCFPasses() {
  // Generated.
  IREE::PCF::registerPasses();
}
} // namespace mlir::iree_compiler
