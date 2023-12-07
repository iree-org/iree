// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Transforms/Passes.h"

#include "iree/compiler/Codegen/WGSL/PassDetail.h"
#include "iree/compiler/Codegen/WGSL/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {
//===---------------------------------------------------------------------===//
// Register WGSL Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/WGSL/Passes.h.inc"
} // namespace

void registerCodegenWGSLPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler
