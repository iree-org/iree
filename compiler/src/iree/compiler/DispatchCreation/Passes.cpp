// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir::iree_compiler;

void DispatchCreation::buildDispatchCreationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  return; // TODO
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/DispatchCreation/Passes.h.inc" // IWYU pragma: keep
} // namespace

void DispatchCreation::registerDispatchCreationPasses() {
  // Generated from Passes.td
  registerPasses();
}
