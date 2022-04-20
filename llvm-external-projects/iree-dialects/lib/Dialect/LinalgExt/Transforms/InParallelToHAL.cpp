// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

static int64_t getNumEnclosingInParallelOps(Operation *op) {
  int64_t numInParallelOps = 0;
  while (auto parentOp = op->getParentOfType<InParallelOp>()) {
    op = parentOp;
    ++numInParallelOps;
  }
  return numInParallelOps;
}

/// Return the unique HALExecutableEntryPointOp within parentFuncOp or creates
/// a new op whose terminator returns the triple (one, one, one).
/// This is a placeholder into which more information can be inserted to build
/// the proper workgroup counts.
/// Return nullptr if the parentFuncOp contains more than a single
/// HALExecutableEntryPointOp.
// TODO: This will not be neded once transform dialect can use real HAL ops.
static HALExecutableEntryPointOp
getOrCreateHALExecutableEntryPointOp(PatternRewriter &rewriter, Location loc,
                                     func::FuncOp parentFuncOp) {
  HALExecutableEntryPointOp entryPointOp;
  WalkResult res = parentFuncOp.walk([&](HALExecutableEntryPointOp op) {
    if (entryPointOp) {
      parentFuncOp.emitError("expected only one executable entry point");
      return WalkResult::interrupt();
    }
    entryPointOp = op;
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return nullptr;
  if (entryPointOp)
    return entryPointOp;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&parentFuncOp.getBody().front());
  entryPointOp = rewriter.create<HALExecutableEntryPointOp>(loc);
  auto region = std::make_unique<Region>();
  Block &block = entryPointOp.workgroup_count_region().emplaceBlock();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    rewriter.create<HALReturnOp>(loc, TypeRange{}, ValueRange{one, one, one});
  }
  return entryPointOp;
}

// TODO: This also needs to do the work of `SetNumWorkgroups` but we can't
// depend on HAL atm.
FailureOr<SmallVector<Operation *>> mlir::iree_compiler::IREE::LinalgExt::
    InParallelOpToHALRewriter::returningMatchAndRewrite(
        iree_compiler::IREE::LinalgExt::InParallelOp inParallelOp,
        PatternRewriter &rewriter) const {
  // TODO: InParallelOp must be nested under a hal variant.
  // We can enable this once we have a proper interface and we split the impl.
  // iree-dialects cannot depend on HAL atm.
  // if (!inParallelOp->getParentOfType<HAL::VariantOp>())
  //   return inParallelOp->emitError("No enclosing HAL::VariantOp");

  // TODO: Ideally only do this on buffers but we can't atm.
  // Bufferize happens at the IREE level on HAL operations, we cannot just
  // call the linalg_transform.bufferize operation here.
  // Instead it happens automatically at the end of the linalg-transform-interp
  // pass.

  // If inParallelOp.num_threads() is already a HAL op, stop applying.
  Operation *numThreadOp = inParallelOp.num_threads().getDefiningOp();
  if (numThreadOp && isa<HALInterfaceWorkgroupIDOp>(numThreadOp))
    return failure();

  // Rewriter-based RAUW operates on Operation* atm, bail if we can't get it.
  Operation *numThreadDefiningOp = inParallelOp.num_threads().getDefiningOp();
  if (!numThreadDefiningOp)
    return failure();

  Location loc = inParallelOp.getLoc();

  // #of enclosing InParallelOp determine the #idx in:
  //   hal.interface.workgroup.id[#idx] : index
  //   hal.interface.workgroup.count[#idx] : index
  unsigned numEnclosingInParallelOps =
      getNumEnclosingInParallelOps(inParallelOp);
  if (numEnclosingInParallelOps >=
      HALExecutableEntryPointOp::getNumWorkgroupDims())
    return failure();

  // Custom hal.executable.entry_point.
  // TODO: getOrCreate at top-level when multiple InParallelOp are used
  // and replace the corresponding return.
  // TODO: pull in the proper dims as the bbArgs for dynamic sizes.
  func::FuncOp enclosingFuncOp = inParallelOp->getParentOfType<func::FuncOp>();
  auto region = std::make_unique<Region>();
  auto entryPointOp =
      getOrCreateHALExecutableEntryPointOp(rewriter, loc, enclosingFuncOp);
  if (!entryPointOp)
    return failure();

  HALReturnOp returnOp = cast<HALReturnOp>(
      entryPointOp.workgroup_count_region().front().getTerminator());
  // Update the numEnclosingInParallelOps^th operand with an in-body clone of
  // numThreadOp.
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(returnOp);
    // TODO: This can only work on constant ops atm. In the future, handle
    // copying full backward slices (but we'll need a better
    // HALExecutableEntryPointOp bbargs contract).
    Operation *op = rewriter.clone(*numThreadOp);
    rewriter.startRootUpdate(returnOp);
    SmallVector<Value> operands = returnOp->getOperands();
    // TODO: ensure this is already 1 or the same value otherwise we are in
    // presence of sibling InParallelOp's that are incompatible.
    operands[numEnclosingInParallelOps] = op->getResult(0);
    returnOp->setOperands(operands);
    rewriter.finalizeRootUpdate(returnOp);
  }

  auto idOp = rewriter.create<HALInterfaceWorkgroupIDOp>(
      loc, numEnclosingInParallelOps);
  auto countOp = rewriter.create<HALInterfaceWorkgroupCountOp>(
      loc, numEnclosingInParallelOps);

  // Get a reference to the terminator that will subsequently be moved.
  PerformConcurrentlyOp performConcurrentlyOp = inParallelOp.getTerminator();

  // First, update the uses of num_threads() within the inParallelOp block.
  rewriter.replaceOpWithinBlock(numThreadDefiningOp, countOp.result(),
                                &inParallelOp.region().front());

  // Steal the iree_compiler::IREE::LinalgExt::InParallel ops, right before the
  // inParallelOp. Replace the bbArg by the HAL id.
  SmallVector<Value> bbArgsTranslated{idOp.result()};
  rewriter.mergeBlockBefore(&inParallelOp.region().front(), inParallelOp,
                            bbArgsTranslated);

  // If we were on buffers, we would be done here.
  if (inParallelOp->getNumResults() == 0) {
    rewriter.eraseOp(inParallelOp);
    return {};
  }

  // On tensors, we need to create sequential insertSlice ops.
  rewriter.setInsertionPoint(inParallelOp);
  SmallVector<Value> results;
  SmallVector<Operation *> resultingOps;
  for (ParallelInsertSliceOp op : performConcurrentlyOp.yieldingOps()) {
    resultingOps.push_back(rewriter.create<tensor::InsertSliceOp>(
        loc, op.source(), op.dest(), op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides()));
    results.push_back(resultingOps.back()->getResult(0));
  }
  rewriter.replaceOp(inParallelOp, results);
  rewriter.eraseOp(performConcurrentlyOp);

  return resultingOps;
}
