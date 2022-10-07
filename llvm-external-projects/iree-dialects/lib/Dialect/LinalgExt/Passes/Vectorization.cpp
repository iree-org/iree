// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

SmallVector<int64_t> interchange(ArrayRef<int64_t> elements,
                                 ArrayRef<int64_t> interchangeVector) {
  SmallVector<int64_t> vec = llvm::to_vector(elements);
  for (auto en : llvm::enumerate(interchangeVector)) {
    vec[en.value()] = elements[en.index()];
  }
  return vec;
}

struct PackOpVectorizationPattern : OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const final {
    if (!packOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(packOp, "require tensor semantics");
    }

    ShapedType resultType = packOp.getOutputType();
    if (!resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(packOp, "require static shapes");
    }

    // TBD: it will be tensor::EmptyOp after an integration.
    // Value dest = rewriter.create<tensor::EmptyOp>(
    // loc, resultType.getShape(), resultType.getElementType());
    Location loc = packOp.getLoc();
    Value dest = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto resultVecType =
        VectorType::get(resultType.getShape(), resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    if (auto pad = packOp.getPaddingValue()) {
      // Initial the empty tensor with padding values.
      Value broadcast =
          rewriter.create<vector::BroadcastOp>(loc, resultVecType, pad);
      SmallVector<Value> indices(resultVecType.getRank(), zero);
      dest = rewriter
                 .create<vector::TransferWriteOp>(
                     loc, broadcast, dest, indices,
                     rewriter.getMultiDimIdentityMap(resultVecType.getRank()))
                 .getResult();
    }

    SmallVector<int64_t> transPerm, innerPosAfterExpansion;
    SmallVector<int64_t> readSizes;
    SmallVector<int64_t> inputVecCastShape;
    ShapedType inputType = packOp.getInputType();
    int64_t inputRank = packOp.getInputRank();
    SmallVector<bool> readInBounds(inputRank, true);
    DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
        packOp.getDimAndTileMapping();
    for (int64_t dim = 0; dim < inputRank; ++dim) {
      if (!tileAndPosMapping.count(dim)) {
        readSizes.push_back(inputType.getDimSize(dim));
        transPerm.push_back(inputVecCastShape.size());
        innerPosAfterExpansion.push_back(-1);
        inputVecCastShape.push_back(inputType.getDimSize(dim));
        continue;
      }

      Optional<int64_t> tileSize =
          getConstantIntValue(tileAndPosMapping.lookup(dim));
      assert(!inputType.isDynamicDim(dim) && tileSize.hasValue() &&
             "something goes really wrong...");
      int64_t size = inputType.getDimSize(dim);
      if (size % tileSize.getValue()) {
        readInBounds[dim] = false;
        readSizes.push_back(llvm::divideCeil(size, tileSize.getValue()) *
                            tileSize.getValue());
      } else {
        readSizes.push_back(size);
      }
      transPerm.push_back(inputVecCastShape.size());
      inputVecCastShape.push_back(readSizes.back() / tileSize.getValue());
      innerPosAfterExpansion.push_back(inputVecCastShape.size());
      inputVecCastShape.push_back(tileSize.getValue());
    }

    for (auto dim : extractFromI64ArrayAttr(packOp.getInnerDimsPos())) {
      transPerm.push_back(innerPosAfterExpansion[dim]);
    }
    if (auto outerDims = packOp.getOuterDimsPerm()) {
      transPerm = interchange(transPerm, extractFromI64ArrayAttr(outerDims));
    }

    auto inputShape = packOp.getInputShape();
    SmallVector<Value> readIndices(
        inputRank, rewriter.create<arith::ConstantIndexOp>(loc, 0));
    auto inputVecType = VectorType::get(readSizes, inputType.getElementType());

    Value read;
    if (packOp.getPaddingValue()) {
      read = rewriter.create<vector::TransferReadOp>(
          loc, inputVecType, packOp.getInput(), readIndices,
          packOp.getPaddingValue(), ArrayRef<bool>{readInBounds});
    } else {
      read = rewriter.create<vector::TransferReadOp>(
          loc, inputVecType, packOp.getInput(), readIndices,
          ArrayRef<bool>{readInBounds});
    }

    Value shapeCast = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get(inputVecCastShape, inputType.getElementType()),
        read);

    Value transpose =
        rewriter.create<vector::TransposeOp>(loc, shapeCast, transPerm);

    SmallVector<bool> writeInBounds(packOp.getOutputRank(), true);
    SmallVector<Value> indices(resultVecType.getRank(), zero);
    Value write =
        rewriter
            .create<vector::TransferWriteOp>(loc, transpose, dest, indices,
                                             ArrayRef<bool>{writeInBounds})
            .getResult();

    rewriter.replaceOp(packOp, write);

    return success();
  }
};

struct LinalgExtVectorizationPass
    : public LinalgExtVectorizationBase<LinalgExtVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, func::FuncDialect, arith::ArithDialect,
                tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PackOpVectorizationPattern>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgExtVectorizationPass() {
  return std::make_unique<LinalgExtVectorizationPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
