// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/StableHLO/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::stablehlo {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/StableHLO/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerStableHLOConversionPasses() {
  // Generated.
  registerPasses();
}

}  // namespace mlir::iree_compiler::stablehlo
