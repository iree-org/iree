// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

namespace IREE::GPU {
namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace
} // namespace IREE::GPU

void registerIREEGPUPasses() {
  // Generated.
  IREE::GPU::registerPasses();
}
} // namespace mlir::iree_compiler
