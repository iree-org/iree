// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements patterns and passes to use `tensor.pad` ops as anchors
// to create separate fast and slow paths inside the kernel. The fast path
// is for inner tiles where we don't need padding, while the slow path is for
// boundary tiles where we do need padding.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-create-fast-slow-path"

namespace mlir::iree_compiler {

/// Returns true if the the given `attrOrValue` is a constant zero.
static bool isZero(OpFoldResult attrOrValue) {
  if (std::optional<int64_t> val = getConstantIntValue(attrOrValue)) {
    return val.value() == 0;
  }
  return false;
}

/// Uses the `tensor.pad` ops as anchors to create separate fast and slow paths
/// inside the kernel. The fast path is for inner tiles where we don't need
/// padding, while the slow path is for boundary tiles where we do need padding.
///
/// This pattern works by creating an `scf.if` op with conditions derived from
/// `tensor.pad` op padding sizes, and copying all ops excluding those for
/// computing padding sizes to both regions of the `scf.if` op.
static void applyFastSlowPathConversion(mlir::FunctionOpInterface funcOp) {
  Block *body = &(*funcOp.getBlocks().begin());

  // Find the anchor tensor.pad op, from which we get the conditions for
  // switching between the fast and slow path.
  auto padOps = llvm::to_vector(body->getOps<tensor::PadOp>());
  if (llvm::size(padOps) != 1)
    return;
  tensor::PadOp padOp = *padOps.begin();

  // If all padding sizes are zero, we don't need to do anything.
  SmallVector<OpFoldResult> lowPads = padOp.getMixedLowPad();
  SmallVector<OpFoldResult> highPads = padOp.getMixedHighPad();
  if (llvm::all_of(lowPads, isZero) && llvm::all_of(highPads, isZero))
    return;

  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPoint(body->getTerminator());
  SmallVector<Operation *, 16> allOps;
  for (Operation &op : body->without_terminator())
    allOps.push_back(&op);

  BackwardSliceOptions options;
  options.filter = [](Operation *op) { return true; };
  SetVector<Operation *> padSizeOps;

  // Build the condition for the scf.if op: all pad sizes are zero.
  Location loc = padOp.getLoc();
  Value cstZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> eqZeroCmpVals;
  for (OpFoldResult pad : llvm::concat<OpFoldResult>(lowPads, highPads)) {
    if (auto padValue = pad.dyn_cast<Value>()) {
      getBackwardSlice(padValue, &padSizeOps, options);
      padSizeOps.insert(padValue.getDefiningOp());
    }
    if (!isZero(pad)) {
      eqZeroCmpVals.push_back(rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          getValueOrCreateConstantIndexOp(rewriter, loc, pad), cstZero));
    }
  }
  Value ifCond = eqZeroCmpVals.front();
  for (Value cmp : llvm::ArrayRef(eqZeroCmpVals).drop_front())
    ifCond = rewriter.create<arith::AndIOp>(loc, ifCond, cmp);

  SmallVector<Operation *> cloneOps;
  for (Operation *op : allOps) {
    if (!padSizeOps.contains(op))
      cloneOps.push_back(op);
  }

  // Build the scf.if op itself. Clone all ops other than those used for
  // computing padding sizes. For the "then" branch, we can elide the padding.
  // For the "else" branch, we retain the clone op.
  auto thenBuilder = [&](OpBuilder &builder, Location loc) {
    IRMapping bvm;
    for (Operation *op : cloneOps) {
      if (op == padOp.getOperation()) {
        // We can elide the tensor.pad op. Just use its source.
        bvm.map(padOp.getResult(), bvm.lookupOrDefault(padOp.getSource()));
      } else {
        builder.clone(*op, bvm);
      }
    }
    builder.create<scf::YieldOp>(loc);
  };
  auto elseBuilder = [&](OpBuilder &builder, Location loc) {
    IRMapping bvm;
    for (Operation *op : cloneOps)
      builder.clone(*op, bvm);
    builder.create<scf::YieldOp>(loc);
  };
  rewriter.create<scf::IfOp>(padOp.getLoc(), ifCond, thenBuilder, elseBuilder);

  // All of these ops have been cloned to both regions. Erease them now.
  for (Operation *op : llvm::reverse(cloneOps))
    rewriter.eraseOp(op);
}

namespace {

struct SPIRVCreateFastSlowPathPass final
    : public SPIRVCreateFastSlowPathBase<SPIRVCreateFastSlowPathPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    applyFastSlowPathConversion(funcOp);

    // Canonicalize the generated scf.if ops. We might have trivially dead
    // branches, in which the sizes might be incorrect due to eliding the
    // tensor.pad op.
    RewritePatternSet patterns(context);
    scf::IfOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSPIRVCreateFastSlowPathPass() {
  return std::make_unique<SPIRVCreateFastSlowPathPass>();
}

} // namespace mlir::iree_compiler
