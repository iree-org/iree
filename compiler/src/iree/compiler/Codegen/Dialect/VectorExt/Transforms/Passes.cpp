// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace IREE::VectorExt {
namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"
} // namespace
} // namespace IREE::VectorExt

void registerIREEVectorExtPasses() {
  // Generated.
  IREE::VectorExt::registerPasses();
}
} // namespace mlir::iree_compiler
