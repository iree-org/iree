// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace detail {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace detail

void registerPasses() { IREE::LinalgExt::detail::registerPasses(); }

} // namespace mlir::iree_compiler::IREE::LinalgExt
