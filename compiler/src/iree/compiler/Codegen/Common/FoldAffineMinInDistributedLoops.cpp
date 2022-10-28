// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldAffineMinInDistributedLoops.cpp --------------------------------===//
//
// This file contains patterns to canonicalize affine.min ops inside tiled and
// distributed loops. These affine.min ops typically take the loop induction
// variable as operands and are used as the size for subtensors/subviews to
// guard against partial tiles. In the case of perfect tiling, they can go away
// to allow exposing static information for vectorization.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fold-affinemin-in-distributed-loops"

namespace mlir {
namespace iree_compiler {

/// Gets the given `attrOrValue` as a Value by creating constant ops for
/// attributes.
static Value getAsValue(OpFoldResult attrOrValue, OpBuilder &builder,
                        Location loc) {
  if (Value val = attrOrValue.dyn_cast<Value>()) return val;
  auto attr = attrOrValue.get<Attribute>().cast<IntegerAttr>();
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

namespace {

/// Folds `affine.min` ops over induction variables of tiled loops that are
/// distributed to processors, where we have the structure:
///
/// ```mlir
/// %id = hal.interface.workgroup.id ...
/// %count = hal.interface.workgroup.count ...
/// %offset = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%id, ...]
/// %size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%count, ...]
/// scf.for %iv = %offset to ... step %size { ... }
/// ```
///
/// For such loops, we need to ignore the distribution aspect and get the
/// lower bound, upper bound, and step for only the tiling aspect, so that we
/// can reuse upstream utilities to prove that the `affine.min` ops are tightly
/// bound so that we can replace them with the tight bound.
struct FoldAffineMinOverDistributedLoopInductionVariable final
    : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override {
    return scf::canonicalizeMinMaxOpInLoop(
        rewriter, minOp, minOp.getAffineMap(), minOp.operands(), /*isMin=*/true,
        matchTileAndDistributedLoop);
  }
};

struct FoldAffineMinInDistributedLoopsPass final
    : public FoldAffineMinInDistributedLoopsBase<
          FoldAffineMinInDistributedLoopsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFoldAffineMinInDistributedLoopsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }
  }
};
}  // namespace

void populateFoldAffineMinInDistributedLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldAffineMinOverDistributedLoopInductionVariable>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
createFoldAffineMinInDistributedLoopsPass() {
  return std::make_unique<FoldAffineMinInDistributedLoopsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
