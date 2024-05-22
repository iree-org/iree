// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-tensor-to-flow"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_DISPATCHTENSORPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

/// Return `true` if the given op is contained in DispatchWorkgroupsOp or in a
/// DispatchRegionOp.
static bool isInDispatchRegion(Operation *op) {
  return op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>() ||
         op->getParentOfType<IREE::Flow::DispatchRegionOp>();
}

/// Wrap a single op in a DispatchWorkgroupsOp.
static FailureOr<IREE::Flow::DispatchWorkgroupsOp>
wrapInWorkgroupsOp(mlir::TensorDimTrackingRewriter &rewriter, Operation *op) {
  // Simplify tensor::DimOps.
  SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
  if (failed(iree_compiler::IREE::Flow::simplifyDimOps(
          rewriter, rewriter.getTensorDimOps())))
    return failure();

  // Wrap operation.
  auto regionOp = IREE::Flow::wrapOpInDispatchRegion(rewriter, op);
  if (failed(regionOp))
    return failure();
  if (failed(cloneProducersToRegion(rewriter, *regionOp)))
    return failure();
  auto workgroupsOp =
      IREE::Flow::rewriteFlowDispatchRegionToFlowDispatchWorkgroups(*regionOp,
                                                                    rewriter);
  if (failed(workgroupsOp))
    return failure();
  return *workgroupsOp;
}

/// Wrap all given ops in a DispatchWorkgroupsOp.
static FailureOr<SmallVector<IREE::Flow::DispatchWorkgroupsOp>>
wrapInWorkgroupsOp(mlir::TensorDimTrackingRewriter &rewriter,
                   SmallVector<Operation *> rootOps) {
  SmallVector<IREE::Flow::DispatchWorkgroupsOp> result;
  for (Operation *rootOp : rootOps) {
    auto workgroupsOp = wrapInWorkgroupsOp(rewriter, rootOp);
    if (failed(workgroupsOp))
      return failure();
    result.push_back(*workgroupsOp);
  }
  return result;
}

/// Wrap all ops of the given types that are direct children of the given op
/// in DispatchWorkgroupsOps.
template <typename... OpTys>
static FailureOr<SmallVector<IREE::Flow::DispatchWorkgroupsOp>>
wrapInWorkgroupsOp(mlir::TensorDimTrackingRewriter &rewriter, Operation *op) {
  // Find ops of type OpTys.
  SmallVector<Operation *> rootOps;
  for (Region &r : op->getRegions())
    for (Block &b : r.getBlocks())
      for (Operation &op : b)
        if (isa<OpTys...>(&op))
          rootOps.push_back(&op);

  // Wrap ops in DispatchWorkgroupsOps.
  return wrapInWorkgroupsOp(rewriter, rootOps);
}

/// Rewrite top-level InsertSliceOps to FlowUpdateOps or wrap them in a
/// dispatch region.
LogicalResult convertInsertSliceOps(
    mlir::TensorDimTrackingRewriter &rewriter, mlir::FunctionOpInterface funcOp,
    SmallVector<IREE::Flow::DispatchWorkgroupsOp> &workgroupsOps) {
  // Find eligible InsertSliceOps.
  SmallVector<tensor::InsertSliceOp> insertSliceOps;
  funcOp.walk([&](tensor::InsertSliceOp op) {
    if (!isInDispatchRegion(op))
      insertSliceOps.push_back(op);
  });

  // Rewrite InsertSliceOps to FlowUpdateOps.
  SmallVector<Operation *> remainingInsertSliceOps;
  for (tensor::InsertSliceOp insertSliceOp : insertSliceOps) {
    if (failed(convertInsertSliceOpToFlowUpdateOp(rewriter, insertSliceOp))) {
      remainingInsertSliceOps.push_back(insertSliceOp);
    }
  }

  // Create a DispatchWorkgroupsOp for every remaining InsertSliceOp.
  FailureOr<SmallVector<IREE::Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingInsertSliceOps);
  if (failed(newWorkgroupsOps))
    return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return success();
}

/// Rewrite top-level ExtractSliceOps to FlowSliceOps or wrap them in a
/// dispatch region.
LogicalResult convertExtractSliceOps(
    mlir::TensorDimTrackingRewriter &rewriter, mlir::FunctionOpInterface funcOp,
    SmallVector<IREE::Flow::DispatchWorkgroupsOp> &workgroupsOps) {
  // Find eligible ExtractSliceOps.
  SmallVector<tensor::ExtractSliceOp> extractSliceOps;
  funcOp.walk([&](tensor::ExtractSliceOp op) {
    if (!isInDispatchRegion(op))
      extractSliceOps.push_back(op);
  });

  // Rewrite ExtractSliceOps to FlowSliceOps.
  SmallVector<Operation *> remainingExtractSliceOps;
  for (tensor::ExtractSliceOp extractSliceOp : extractSliceOps) {
    if (failed(convertExtractSliceOpToFlowSliceOp(rewriter, extractSliceOp))) {
      remainingExtractSliceOps.push_back(extractSliceOp);
    }
  }

  // Create a DispatchWorkgroupsOp for every remaining ExtractSliceOp.
  FailureOr<SmallVector<IREE::Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingExtractSliceOps);
  if (failed(newWorkgroupsOps))
    return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return success();
}

namespace {
struct DispatchTensorPass
    : public IREE::Flow::impl::DispatchTensorPassBase<DispatchTensorPass> {
  using IREE::Flow::impl::DispatchTensorPassBase<
      DispatchTensorPass>::DispatchTensorPassBase;
  void runOnOperation() override;
};
} // namespace

void DispatchTensorPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  mlir::TensorDimTrackingRewriter rewriter(funcOp);
  mlir::MLIRContext *context = &getContext();

  auto workgroupsOps = SmallVector<IREE::Flow::DispatchWorkgroupsOp>();
  funcOp->walk([&](Flow::DispatchWorkgroupsOp workgroupsOp) {
    workgroupsOps.push_back(workgroupsOp);
  });

  // Rewrite InsertSliceOps to FlowUpdateOps.
  if (failed(convertInsertSliceOps(rewriter, funcOp, workgroupsOps))) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.insert_slice`");
    return signalPassFailure();
  }

  // Rewrite ExtractSliceOps to FlowUpdateOps.
  if (failed(convertExtractSliceOps(rewriter, funcOp, workgroupsOps))) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.extract_slice`");
    return signalPassFailure();
  }

  // Canonicalize to flow.tensor ops.
  RewritePatternSet convertToFlowPatterns(context);
  IREE::Flow::populateTensorToFlowConversionPatterns(context,
                                                     convertToFlowPatterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(
      convertToFlowPatterns);
  IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(
      convertToFlowPatterns, context);
  IREE::Flow::TensorBitCastOp::getCanonicalizationPatterns(
      convertToFlowPatterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp,
                                          std::move(convertToFlowPatterns)))) {
    funcOp->emitOpError("failed conversion to flow.tensor ops");
    return signalPassFailure();
  }

  // fold `tensor.insert_slice/extract_slice` operations with
  // `flow.dispatch.tensor.load/store`.
  RewritePatternSet foldExtractInsertSliceOps(context);
  IREE::Flow::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
      foldExtractInsertSliceOps, context);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(foldExtractInsertSliceOps)))) {
    funcOp->emitOpError("failed to insert/extract_slice with "
                        "flow.dispatch.tensor.load/store");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
