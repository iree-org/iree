// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-remove-trivial-loops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REMOVESINGLEITERATIONLOOPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static LogicalResult removeOneTripTiledLoops(mlir::FunctionOpInterface funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns);
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveSingleIterationLoopPass final
    : public impl::RemoveSingleIterationLoopPassBase<
          RemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    if (failed(removeOneTripTiledLoops(funcOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
