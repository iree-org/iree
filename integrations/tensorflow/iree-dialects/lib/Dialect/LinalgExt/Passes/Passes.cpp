// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace detail {
#define GEN_PASS_REGISTRATION
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h.inc"  // IWYU pragma: export
}  // namespace detail

}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

void IREE::LinalgExt::registerPasses() {
  IREE::LinalgExt::detail::registerPasses();
}
