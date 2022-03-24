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

static constexpr auto kFakeHALEntryPointRegionOpName =
    "fake_hal_entry_point_region";
static constexpr auto kFakeHALReturnOpName = "fake_hal_return";
static constexpr auto kFakeHALWorkgroupIdOpName = "fake_hal_workgroup_id";
static constexpr auto kFakeHALWorkgroupCountOpName = "fake_hal_workgroup_count";

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

  // TODO: #of enclosing InParallelOp determine the #idx in:
  //   hal.interface.workgroup.id[#idx] : index
  //   hal.interface.workgroup.count[#idx] : index
  unsigned numEnclosingInParallelOps = 0;

  // If inParallelOp.num_threads() is already a HAL op, stop applying.
  Operation *numThreadOp = inParallelOp.num_threads().getDefiningOp();
  if (numThreadOp &&
      numThreadOp->getName().getStringRef() == kFakeHALWorkgroupIdOpName)
    return failure();

  // Rewriter-based RAUW operates on Operation* atm, bail if we can't get it.
  Operation *numThreadDefiningOp = inParallelOp.num_threads().getDefiningOp();
  if (!numThreadDefiningOp)
    return failure();

  Location loc = inParallelOp.getLoc();

  // Custom hal.executable.entry_point.
  auto region = std::make_unique<Region>();
  Block &block = region->emplaceBlock();
  {
    // TODO: getOrCreate at top-level when multiple InParallelOp are used and
    // replace the corresponding return.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Operation *op = rewriter.clone(*numThreadOp);
    OperationState entryPointReturnOpState(
        loc, kFakeHALReturnOpName, ValueRange{op->getResult(0), one, one},
        TypeRange{}, ArrayRef<NamedAttribute>{}, BlockRange{}, {});
    rewriter.create(entryPointReturnOpState);
  }
  OperationState entryPointOpState(
      loc, kFakeHALEntryPointRegionOpName, ValueRange{}, TypeRange{},
      ArrayRef<NamedAttribute>{}, BlockRange{}, {region});
  Operation *entryPointOp = rewriter.create(entryPointOpState);

  // Custom hal.workgroup.id.
  OperationState idOpState(
      loc, kFakeHALWorkgroupIdOpName, ValueRange{},
      TypeRange{rewriter.getIndexType()},
      ArrayRef<NamedAttribute>{
          NamedAttribute(rewriter.getStringAttr("idx"),
                         rewriter.getIndexAttr(numEnclosingInParallelOps))});
  Operation *idOp = rewriter.create(idOpState);

  // Custom hal.workgroup.count.
  OperationState countOpState(
      loc, kFakeHALWorkgroupCountOpName, ValueRange{},
      TypeRange{rewriter.getIndexType()},
      ArrayRef<NamedAttribute>{
          NamedAttribute(rewriter.getStringAttr("idx"),
                         rewriter.getIndexAttr(numEnclosingInParallelOps))});
  Operation *countOp = rewriter.create(countOpState);

  // Get a reference to the terminator that will subsequently be moved.
  PerformConcurrentlyOp performConcurrentlyOp = inParallelOp.getTerminator();

  // First, update the uses of num_threads() within the inParallelOp block.
  rewriter.replaceOpWithinBlock(numThreadDefiningOp, countOp->getResult(0),
                                &inParallelOp.region().front());

  // Steal the iree_compiler::IREE::LinalgExt::InParallel ops, right before the
  // inParallelOp. Replace the bbArg by the HAL id.
  SmallVector<Value> bbArgsTranslated{idOp->getResult(0)};
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
