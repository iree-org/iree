// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerParametersPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::IREE::IO::Parameters
