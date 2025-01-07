// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertAccGEMMtoGEMMpass.cpp ----------------------------------===//
//
// Converts Accumulating GEMM to GEMM + elementwise add.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTACCGEMMTOGEMMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct ConvertAccGEMMtoGEMM
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(linalgOp) &&
        !isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
      return failure();
    }
    if (!linalgOp.hasPureTensorSemantics())
      return failure();

    // Nothing to do if the output tensor operand is already a fill op.
    SmallVector<OpOperand *> outputOperands;
    if (!linalgOp.hasPureBufferSemantics()) {
      outputOperands = llvm::to_vector(
          llvm::map_range(linalgOp.getDpsInitsMutable(),
                          [](OpOperand &opOperand) { return &opOperand; }));
    }

    Value outputOperand = outputOperands.front()->get();

    auto outsDefiningOp =
        outputOperand.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!outsDefiningOp) {
      // If not DispatchTensorLoadOp then do nothing.
      return failure();
    }
    auto outputType = llvm::cast<RankedTensorType>(outputOperand.getType());
    if (!outputType.getElementType().isIntOrFloat())
      return failure();
    auto elementType = outputType.getElementType();

    Location loc = linalgOp.getLoc();

    // Check if the output tensor access is a projected permutation
    if (!linalgOp.getMatchingIndexingMap(outputOperands.front())
             .isProjectedPermutation()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Output indexing map must be a projected permutation.");
    }

    int64_t outputRank = outputType.getRank();
    SmallVector<utils::IteratorType> iterators(outputRank,
                                               utils::IteratorType::parallel);
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(outputRank));

    // Create a zero tensor as the new output tensor operand to the Linalg
    // contraction op.
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, outputOperand);
    auto initOp =
        rewriter.create<tensor::EmptyOp>(loc, mixedSizes, elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value fill =
        rewriter.create<linalg::FillOp>(loc, zero, initOp.getResult()).result();

    // Update the contraction op to use the new zero tensor as output operand.
    rewriter.modifyOpInPlace(linalgOp,
                             [&]() { linalgOp.setDpsInitOperand(0, fill); });

    // Create a generic op to add back the original output tensor operand.
    rewriter.setInsertionPointAfter(linalgOp);
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{linalgOp->getResult(0), outputOperand},
        fill, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result;
          if (llvm::isa<FloatType>(elementType)) {
            result = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          } else {
            result = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
          }
          b.create<linalg::YieldOp>(nestedLoc, result);
        });
    linalgOp->getResult(0).replaceAllUsesExcept(genericOp->getResult(0),
                                                genericOp);
    return success();
  }
};

struct ConvertAccGEMMToGEMMPass
    : public impl::ConvertAccGEMMToGEMMPassBase<ConvertAccGEMMToGEMMPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAccGEMMtoGEMM>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler
