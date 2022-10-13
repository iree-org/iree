// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
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

Value getInputOrPaddedInput(OpBuilder &builder, PackOp packOp) {
  Value input = packOp.getInput();
  if (!packOp.getPaddingValue()) {
    return input;
  }

  Location loc = packOp.getLoc();
  SmallVector<OpFoldResult> lowPadding, highPadding;
  auto zeroAttr = builder.getIndexAttr(0);
  ShapedType inputType = packOp.getInputType();
  int64_t inputRank = inputType.getRank();
  lowPadding.append(inputRank, zeroAttr);

  SmallVector<int64_t> paddedShape;
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
      packOp.getDimAndTileMapping();
  for (int64_t dim = 0; dim < inputRank; ++dim) {
    int64_t size = inputType.getDimSize(dim);
    if (!tileAndPosMapping.count(dim)) {
      paddedShape.push_back(size);
      highPadding.push_back(zeroAttr);
      continue;
    }

    Optional<int64_t> tileSize =
        getConstantIntValue(tileAndPosMapping.lookup(dim));
    assert(!inputType.isDynamicDim(dim) && tileSize.hasValue() &&
           "something goes really wrong...");
    int64_t sizeWithPad = llvm::alignTo(size, tileSize.getValue());
    highPadding.push_back(builder.getIndexAttr(sizeWithPad - size));
    paddedShape.push_back(sizeWithPad);
  }
  auto resultType =
      RankedTensorType::get(paddedShape, inputType.getElementType());
  return tensor::createPadScalarOp(resultType, input, packOp.getPaddingValue(),
                                   lowPadding, highPadding,
                                   /*nofold=*/false, loc, builder);
}

// Creates a linalg.generic that transposes input using permutation indices.
// Example: (M1, M0, N1, N0) -> (M1, N1, M0, N0) if indices = {0, 2, 1, 3}.
Value createTransposeOp(Location loc, OpBuilder &builder, Value input,
                        ArrayRef<int64_t> indices) {
  auto inputType = input.getType().cast<RankedTensorType>();
  auto nloops = indices.size();

  SmallVector<AffineExpr> exprs = llvm::to_vector<4>(
      llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
        return builder.getAffineDimExpr(index);
      }));

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<OpFoldResult> targetShape;
  for (auto i : indices) {
    if (inputShape[i] == ShapedType::kDynamicSize) {
      targetShape.emplace_back(builder.create<tensor::DimOp>(loc, input, i));
    } else {
      targetShape.push_back(builder.getIndexAttr(inputShape[i]));
    }
  }

  Value outputTensor = builder.create<tensor::EmptyOp>(
      loc, targetShape, inputType.getElementType());

  SmallVector<StringRef, 4> loopAttributeTypes(nloops,
                                               getParallelIteratorTypeName());

  SmallVector<AffineMap, 2> indexingMaps = {
      inversePermutation(
          AffineMap::get(nloops, 0, exprs, builder.getContext())),
      AffineMap::getMultiDimIdentityMap(nloops, builder.getContext())};

  auto transposedOp = builder.create<linalg::GenericOp>(
      loc, outputTensor.getType(),
      /*inputs=*/input, /*outputs=*/outputTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return transposedOp.getResult(0);
}

// Rewrites iree_linalg_ext.pack to tensor.pad + tensor.expand_shape +
// linalg.generic (transpose) ops.
struct GeneralizePackOpPattern : OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const final {
    if (!packOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(packOp, "require tensor semantics");
    }

    Value input = getInputOrPaddedInput(rewriter, packOp);

    SmallVector<ReassociationIndices> expandIndices;
    SmallVector<int64_t> transPerm, innerPosAfterExpansion;
    SmallVector<int64_t> inputVecCastShape;
    ShapedType paddedInputType = input.getType();
    int64_t inputRank = packOp.getInputRank();
    SmallVector<bool> readInBounds(inputRank, true);
    DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
        packOp.getDimAndTileMapping();
    for (int64_t dim = 0, idx = 0; dim < inputRank; ++dim) {
      int64_t dimSize = paddedInputType.getDimSize(dim);
      if (!tileAndPosMapping.count(dim)) {
        transPerm.push_back(inputVecCastShape.size());
        innerPosAfterExpansion.push_back(-1);
        inputVecCastShape.push_back(dimSize);
        expandIndices.push_back(ReassociationIndices{idx});
        idx++;
        continue;
      }

      Optional<int64_t> tileSize =
          getConstantIntValue(tileAndPosMapping.lookup(dim));
      transPerm.push_back(inputVecCastShape.size());
      inputVecCastShape.push_back(dimSize / tileSize.getValue());
      innerPosAfterExpansion.push_back(inputVecCastShape.size());
      inputVecCastShape.push_back(tileSize.getValue());
      expandIndices.push_back(ReassociationIndices{idx, idx + 1});
      idx += 2;
    }

    for (auto dim : extractFromI64ArrayAttr(packOp.getInnerDimsPos())) {
      transPerm.push_back(innerPosAfterExpansion[dim]);
    }
    if (auto outerDims = packOp.getOuterDimsPerm()) {
      transPerm = interchange(transPerm, extractFromI64ArrayAttr(outerDims));
    }

    Location loc = packOp.getLoc();
    auto expandTargetType = RankedTensorType::get(
        inputVecCastShape, paddedInputType.getElementType());
    auto expand = rewriter.create<tensor::ExpandShapeOp>(loc, expandTargetType,
                                                         input, expandIndices);
    auto trans = createTransposeOp(loc, rewriter, expand, transPerm);
    rewriter.replaceOp(packOp, trans);
    return success();

  }
};

struct ExpandShapeVectorizationPattern
    : OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExpandShapeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto srcShapedType = op.getSrc().getType().cast<ShapedType>();
    auto inputVecType =
        VectorType::get(srcShapedType.getShape(), srcShapedType.getElementType());
    SmallVector<Value> readIndices(
        srcShapedType.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
    SmallVector<bool> readInBounds(srcShapedType.getRank(), true);
    Value read = rewriter.create<vector::TransferReadOp>(
        loc, inputVecType, op.getSrc(), readIndices,
        ArrayRef<bool>{readInBounds});

    auto destShapedType = op.getResult().getType().cast<ShapedType>();
    auto destVecType = VectorType::get(destShapedType.getShape(),
                                       destShapedType.getElementType());
    Value shapeCast =
        rewriter.create<vector::ShapeCastOp>(loc, destVecType, read);

    // TBD: it will be tensor::EmptyOp after an integration.
    SmallVector<Value> writeIndices(
        destShapedType.getRank(),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));
    Value dest = rewriter.create<tensor::EmptyOp>(
        loc, destShapedType.getShape(), destShapedType.getElementType());

    Value write = rewriter.create<vector::TransferWriteOp>(
        loc, shapeCast, dest, writeIndices,
        rewriter.getMultiDimIdentityMap(destShapedType.getRank())).getResult();
    rewriter.replaceOp(op, write);

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
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<GeneralizePackOpPattern>(ctx);
    patterns.add<ExpandShapeVectorizationPattern, LinalgVectorizationPattern>(
        ctx);
    linalg::populatePadOpVectorizationPatterns(patterns);

    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
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
