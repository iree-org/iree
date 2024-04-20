// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CONVERTTOFLOWPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

// Pass to test conversion to flow patterns.
struct ConvertToFlowPass
    : public IREE::Flow::impl::ConvertToFlowPassBase<ConvertToFlowPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet convertToFlowPatterns(context);
    IREE::Flow::populateTensorToFlowConversionPatterns(context,
                                                       convertToFlowPatterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        convertToFlowPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(convertToFlowPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
