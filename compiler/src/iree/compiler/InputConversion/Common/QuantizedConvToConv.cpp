// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/InputConversion/Common/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

Value makeLinalgInit(ImplicitLocOpBuilder &rewriter, Value value) {
  RankedTensorType ty = value.getType().cast<RankedTensorType>();
  Type eTy = ty.getElementType();

  SmallVector<Value> dynSizes;
  for (int i = 0, s = ty.getRank(); i < s; i++) {
    if (ty.isDynamicDim(i)) {
      dynSizes.push_back(rewriter.create<tensor::DimOp>(value, i));
    }
  }

  return rewriter.create<tensor::EmptyOp>(ty.getShape(), eTy, dynSizes);
}

Value applyZeroPoint(ImplicitLocOpBuilder &rewriter, Value conv, Value sum,
                     Value zp, ArrayRef<int> affine_map) {
  auto context = rewriter.getContext();
  auto convTy = conv.getType().cast<RankedTensorType>();

  llvm::SmallVector<AffineExpr> sumExprs;
  for (auto i : affine_map) {
    sumExprs.push_back(rewriter.getAffineDimExpr(i));
  }

  SmallVector<utils::IteratorType> iterators(convTy.getRank(),
                                             utils::IteratorType::parallel);

  auto convMap = rewriter.getMultiDimIdentityMap(convTy.getRank());
  auto sumMap = AffineMap::get(convTy.getRank(), 0, sumExprs, context);

  SmallVector<AffineMap> affineMaps{convMap, sumMap, convMap};

  Value init = makeLinalgInit(rewriter, conv);
  return rewriter
      .create<linalg::GenericOp>(
          init.getType(), ValueRange{conv, sum}, ValueRange{init}, affineMaps,
          iterators,
          [=](OpBuilder &b, Location loc, ValueRange args) {
            Value mul = b.create<arith::MulIOp>(loc, args[1], zp);
            Value sum = b.create<arith::SubIOp>(loc, args[0], mul);
            b.create<linalg::YieldOp>(loc, sum);
          })
      .getResult(0);
}

// Pattern lowering conv_2d_nhwc_hwcf_q to conv_2d_nhwc_hwcf.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
struct QuantizedConvToConv
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfQOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfQOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto input = op.getInputs()[0];
    auto filter = op.getInputs()[1];
    auto iZp = op.getInputs()[2];
    auto fZp = op.getInputs()[3];
    auto inputTy = input.getType().cast<RankedTensorType>();
    auto filterTy = filter.getType().cast<RankedTensorType>();
    auto resultTy = op.getType(0).cast<ShapedType>();
    auto accETy = resultTy.getElementType();

    auto strides = op.getStrides();
    auto dilations = op.getDilations();

    IntegerAttr iZpConst;
    IntegerAttr fZpConst;
    bool iZpIsZero = matchPattern(iZp, m_Constant(&iZpConst)) &&
                     iZpConst.getValue().isZero();
    bool fZpIsZero = matchPattern(fZp, m_Constant(&fZpConst)) &&
                     fZpConst.getValue().isZero();

    // First implement the convolution without the zero point.
    Value newConv = builder
                        .create<linalg::Conv2DNhwcHwcfOp>(
                            resultTy, ValueRange{input, filter},
                            op.getOutputs(), strides, dilations)
                        .getResult(0);
    RankedTensorType newConvTy = newConv.getType().cast<RankedTensorType>();

    // If the zero pointer value is zero we can just replace.
    if (iZpIsZero && fZpIsZero) {
      rewriter.replaceOp(op, newConv);
      return success();
    }

    // Compute the summation and correction for the filter zero point:
    //  sum(d) = filter(a, b, c, d)
    //  conv(a, b, c, d) -= iZp * filter(d)
    if (!iZpIsZero) {
      Value filterSum =
          sumReduceDimensionSubset(builder, filter, accETy,
                                   /*reduce_dim=*/{true, true, true, false});
      newConv = applyZeroPoint(builder, newConv, filterSum, iZp, {3});
    }

    if (!fZpIsZero) {
      // Reduce along the input feature dimension:
      //   sum(a, b, c) = input(a, b, c, d)
      Value inputSum =
          sumReduceDimensionSubset(builder, input, accETy,
                                   /*reduce_dim*/ {false, false, false, true});

      // Materialize a length-1 dimension at the end of the summation.
      SmallVector<ReassociationExprs, 4> reassociationMap(3);
      for (int i = 0; i < 3; i++)
        reassociationMap[i].push_back(builder.getAffineDimExpr(i));
      reassociationMap.back().push_back(builder.getAffineDimExpr(3));

      auto expandTy =
          RankedTensorType::get({inputTy.getDimSize(0), inputTy.getDimSize(1),
                                 inputTy.getDimSize(2), 1},
                                accETy);
      inputSum = builder.create<tensor::ExpandShapeOp>(expandTy, inputSum,
                                                       reassociationMap);

      // Perform a sum-pooling operation across the kernel width and height.
      auto poolTy = RankedTensorType::get(
          {expandTy.getDimSize(0), newConvTy.getDimSize(1),
           newConvTy.getDimSize(2), expandTy.getDimSize(3)},
          accETy);

      llvm::SmallVector<Value> poolDynDims;
      if (expandTy.isDynamicDim(0))
        poolDynDims.push_back(builder.create<tensor::DimOp>(inputSum, 0));

      if (newConvTy.isDynamicDim(1))
        poolDynDims.push_back(builder.create<tensor::DimOp>(newConv, 1));

      if (newConvTy.isDynamicDim(2))
        poolDynDims.push_back(builder.create<tensor::DimOp>(newConv, 2));

      if (expandTy.isDynamicDim(3))
        poolDynDims.push_back(builder.create<tensor::DimOp>(inputSum, 3));

      Value poolTensor = builder.create<tensor::EmptyOp>(poolTy.getShape(),
                                                         accETy, poolDynDims);

      Attribute initialAttr = builder.getZeroAttr(accETy);
      Value initialValue = builder.create<arith::ConstantOp>(initialAttr);
      poolTensor = builder
                       .create<linalg::FillOp>(ValueRange{initialValue},
                                               ValueRange{poolTensor})
                       .result();

      llvm::SmallVector<int64_t> kernel{filterTy.getDimSize(0),
                                        filterTy.getDimSize(1)};
      llvm::SmallVector<Value> kernelDynDims;
      for (int i = 0; i < 2; i++) {
        if (filterTy.isDynamicDim(i))
          kernelDynDims.push_back(builder.create<tensor::DimOp>(filter, i));
      }

      Value poolDims =
          builder.create<tensor::EmptyOp>(kernel, accETy, kernelDynDims);
      inputSum = builder
                     .create<linalg::PoolingNhwcSumOp>(
                         ArrayRef<Type>{poolTy}, ValueRange{inputSum, poolDims},
                         poolTensor, strides, dilations)
                     .getResult(0);

      // Collapse the length-1 ending dimension away.
      auto collapseTy =
          RankedTensorType::get(poolTy.getShape().drop_back(), accETy);
      inputSum = builder.create<tensor::CollapseShapeOp>(collapseTy, inputSum,
                                                         reassociationMap);

      // Apply the zero-point update based on the input sum.
      newConv = applyZeroPoint(builder, newConv, inputSum, fZp, {0, 1, 2});
    }

    // Apply the final update that occurs when there are multiple zero-points.
    if (!iZpIsZero && !fZpIsZero) {
      Value dim0 = builder.create<tensor::DimOp>(filter, 0);
      Value dim1 = builder.create<tensor::DimOp>(filter, 1);
      Value dim2 = builder.create<tensor::DimOp>(filter, 2);
      Value mul_dim0_dim1 = builder.create<arith::MulIOp>(dim0, dim1);
      Value mul_dim0_dim1_dim2 =
          builder.create<arith::MulIOp>(mul_dim0_dim1, dim2);
      Value cast =
          builder.create<arith::IndexCastOp>(accETy, mul_dim0_dim1_dim2);

      auto convTy = newConv.getType().cast<RankedTensorType>();
      SmallVector<utils::IteratorType> iterators(convTy.getRank(),
                                                 utils::IteratorType::parallel);
      auto convMap = rewriter.getMultiDimIdentityMap(convTy.getRank());
      SmallVector<AffineMap> affineMaps{convMap, convMap};

      Value init = makeLinalgInit(builder, newConv);
      newConv = builder
                    .create<linalg::GenericOp>(
                        init.getType(), ValueRange{newConv}, ValueRange{init},
                        affineMaps, iterators,
                        [=](OpBuilder &b, Location loc, ValueRange args) {
                          Value mul1 = b.create<arith::MulIOp>(loc, iZp, fZp);
                          Value mul2 = b.create<arith::MulIOp>(loc, mul1, cast);
                          Value sum =
                              b.create<arith::AddIOp>(loc, args[0], mul2);
                          b.create<linalg::YieldOp>(loc, sum);
                        })
                    .getResult(0);
    }

    rewriter.replaceOp(op, newConv);
    return success();
  }
};

/// Pass that lowers quantized_conv to conv.
struct LinalgQuantizedConvToConvPass
    : public LinalgQuantizedConvToConvPassBase<LinalgQuantizedConvToConvPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    patterns.add<QuantizedConvToConv>(context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgQuantizedConvToConvPass() {
  return std::make_unique<LinalgQuantizedConvToConvPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
