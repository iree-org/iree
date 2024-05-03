// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-dispatch-workgroups-cleanup"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_DISPATCHWORKGROUPSCLEANUPPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

struct DispatchWorkgroupsCleanupPass
    : public IREE::Flow::impl::DispatchWorkgroupsCleanupPassBase<
          DispatchWorkgroupsCleanupPass>::DispatchWorkgroupsCleanupPassBase {
  using IREE::Flow::impl::DispatchWorkgroupsCleanupPassBase<
      DispatchWorkgroupsCleanupPass>::DispatchWorkgroupsCleanupPassBase;
  void runOnOperation() override;
};

void DispatchWorkgroupsCleanupPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  mlir::TensorDimTrackingRewriter rewriter(funcOp);
  // A few extra canonicalizations/lowerings.
  {
    RewritePatternSet convertToFlowPatterns(context);
    IREE::Flow::populateTensorToFlowConversionPatterns(context,
                                                       convertToFlowPatterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        convertToFlowPatterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(
        convertToFlowPatterns, context);
    IREE::Flow::TensorBitCastOp::getCanonicalizationPatterns(
        convertToFlowPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(convertToFlowPatterns)))) {
      funcOp->emitOpError("failed conversion to flow.tensor ops");
      return signalPassFailure();
    }

    // Finally fold `tensor.insert_slice/extract_slice` operations with
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
}

} // namespace mlir::iree_compiler::IREE::Flow
