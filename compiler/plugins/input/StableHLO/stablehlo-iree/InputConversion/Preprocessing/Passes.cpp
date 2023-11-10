// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "stablehlo-iree/InputConversion/Preprocessing/Passes.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
#define GEN_PASS_REGISTRATION
#include "stablehlo-iree/InputConversion/Preprocessing/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerStableHLOPreprocessingPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::stablehlo
