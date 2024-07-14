// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-tensor-to-flow"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CONVERTTENSORTOFLOWPASS
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

/// Rewrite top-level InsertSliceOps to FlowUpdateOps or wrap them in a
/// dispatch region. Returns the number of dispatches for non-contiguous insert
/// slices created.
static FailureOr<int> convertInsertSliceOps(
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
  int64_t numRemainingInsertSliceOps =
      static_cast<int64_t>(remainingInsertSliceOps.size());

  // Create a DispatchWorkgroupsOp for every remaining InsertSliceOp.
  FailureOr<SmallVector<IREE::Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingInsertSliceOps);
  if (failed(newWorkgroupsOps))
    return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return numRemainingInsertSliceOps;
}

/// Rewrite top-level ExtractSliceOps to FlowSliceOps or wrap them in a
/// dispatch region. Returns the number of dispatches for non-contiguous extract
/// slices created.
static FailureOr<size_t> convertExtractSliceOps(
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

  int64_t numRemainingExtractSliceOps =
      static_cast<int64_t>(remainingExtractSliceOps.size());

  // Create a DispatchWorkgroupsOp for every remaining ExtractSliceOp.
  FailureOr<SmallVector<IREE::Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingExtractSliceOps);
  if (failed(newWorkgroupsOps))
    return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return numRemainingExtractSliceOps;
}

namespace {
struct ConvertTensorToFlowPass
    : public IREE::Flow::impl::ConvertTensorToFlowPassBase<
          ConvertTensorToFlowPass> {
  using IREE::Flow::impl::ConvertTensorToFlowPassBase<
      ConvertTensorToFlowPass>::ConvertTensorToFlowPassBase;
  void runOnOperation() override;
};
} // namespace

void ConvertTensorToFlowPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  mlir::TensorDimTrackingRewriter rewriter(funcOp);
  mlir::MLIRContext *context = &getContext();

  auto workgroupsOps = SmallVector<IREE::Flow::DispatchWorkgroupsOp>();
  funcOp->walk([&](Flow::DispatchWorkgroupsOp workgroupsOp) {
    workgroupsOps.push_back(workgroupsOp);
  });

  // Rewrite InsertSliceOps to FlowUpdateOps.
  FailureOr<size_t> numSlowInsertSliceDispatches =
      convertInsertSliceOps(rewriter, funcOp, workgroupsOps);
  if (failed(numSlowInsertSliceDispatches)) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.insert_slice`");
    return signalPassFailure();
  }
  numSlowCopyDispatches += numSlowInsertSliceDispatches.value();

  // Rewrite ExtractSliceOps to FlowUpdateOps.
  FailureOr<size_t> numSlowExtractSliceDispatches =
      convertExtractSliceOps(rewriter, funcOp, workgroupsOps);
  if (failed(numSlowExtractSliceDispatches)) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.extract_slice`");
    return signalPassFailure();
  }
  numSlowCopyDispatches += numSlowExtractSliceDispatches.value();

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
