// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_LINALGQUANTIZEDCONVTOCONVPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// Creates an empty copy matching the provided value.
Value emptyCopy(ImplicitLocOpBuilder &rewriter, Value value) {
  Type eTy = getElementTypeOrSelf(value.getType());
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, rewriter.getLoc(), value);
  return tensor::EmptyOp::create(rewriter, mixedSizes, eTy);
}

// Creates an zero initialized tensor of given shape and type.
Value emptyZero(ImplicitLocOpBuilder &builder, RankedTensorType ty,
                llvm::SmallVector<Value> dyn) {
  Value empty =
      tensor::EmptyOp::create(builder, ty.getShape(), ty.getElementType(), dyn);

  TypedAttr attr = builder.getZeroAttr(ty.getElementType());
  Value cnst = arith::ConstantOp::create(builder, attr);
  return linalg::FillOp::create(builder, ValueRange{cnst}, ValueRange{empty})
      .result();
}

// Apply the multiply subtract corresponding with a zero-point adjustment
// broadcasting according to the afifne map.
Value applyZeroPoint(ImplicitLocOpBuilder &builder, Value conv, Value sum,
                     Value zp, ArrayRef<int> affine_map) {
  auto context = builder.getContext();
  auto convTy = cast<RankedTensorType>(conv.getType());

  llvm::SmallVector<AffineExpr> sumExprs;
  for (auto i : affine_map) {
    sumExprs.push_back(builder.getAffineDimExpr(i));
  }

  SmallVector<utils::IteratorType> iterators(convTy.getRank(),
                                             utils::IteratorType::parallel);

  auto convMap = builder.getMultiDimIdentityMap(convTy.getRank());
  auto sumMap = AffineMap::get(convTy.getRank(), 0, sumExprs, context);

  SmallVector<AffineMap> affineMaps{convMap, sumMap, convMap};

  Value init = emptyCopy(builder, conv);
  return linalg::GenericOp::create(
             builder, init.getType(), ValueRange{conv, sum}, ValueRange{init},
             affineMaps, iterators,
             [=](OpBuilder &b, Location loc, ValueRange args) {
               Value mul = arith::MulIOp::create(b, loc, args[1], zp);
               Value sum = arith::SubIOp::create(b, loc, args[0], mul);
               linalg::YieldOp::create(b, loc, sum);
             })
      .getResult(0);
}

// Add the scalar value to the tensor.
Value addScalar(ImplicitLocOpBuilder &builder, Value value, Value scalar) {
  auto ty = cast<RankedTensorType>(value.getType());
  SmallVector<utils::IteratorType> iterators(ty.getRank(),
                                             utils::IteratorType::parallel);
  auto map = builder.getMultiDimIdentityMap(ty.getRank());
  Value init = emptyCopy(builder, value);
  return linalg::GenericOp::create(
             builder, init.getType(), ValueRange{value}, ValueRange{init},
             ArrayRef<AffineMap>{map, map}, iterators,
             [=](OpBuilder &b, Location loc, ValueRange args) {
               Value add = arith::AddIOp::create(b, loc, args[0], scalar);
               linalg::YieldOp::create(b, loc, add);
             })
      .getResult(0);
}

void GetDynamicDym(ImplicitLocOpBuilder &builder,
                   llvm::SmallVector<int64_t> &dims,
                   llvm::SmallVector<Value> &dynDims, Value value,
                   int64_t dim) {
  ShapedType ty = cast<ShapedType>(value.getType());
  dims.push_back(ty.getDimSize(dim));
  if (ty && ty.isDynamicDim(dim)) {
    dynDims.push_back(tensor::DimOp::create(builder, value, dim));
  }
}

Value multiplyDims(ImplicitLocOpBuilder &builder, Value value,
                   llvm::ArrayRef<int64_t> dims) {
  Value count = tensor::DimOp::create(builder, value, dims.front());

  for (auto d : dims.drop_front()) {
    Value dim = tensor::DimOp::create(builder, value, d);
    count = arith::MulIOp::create(builder, count, dim);
  }

  return count;
}

// Pattern lowering conv_2d_nhwc_hwcf_q to conv_2d_nhwc_hwcf.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
struct QuantizedConvToConv
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfQOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfQOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto input = op.getInputs()[0];
    auto filter = op.getInputs()[1];
    auto iZp = op.getInputs()[2];
    auto fZp = op.getInputs()[3];
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto resultTy = cast<ShapedType>(op.getType(0));
    auto accETy = resultTy.getElementType();

    auto strides = op.getStrides();
    auto dilations = op.getDilations();

    IntegerAttr iZpConst, fZpConst;
    bool iZpIsZero = matchPattern(iZp, m_Constant(&iZpConst)) &&
                     iZpConst.getValue().isZero();
    bool fZpIsZero = matchPattern(fZp, m_Constant(&fZpConst)) &&
                     fZpConst.getValue().isZero();

    // First implement the convolution without the zero point.
    Value newConv = linalg::Conv2DNhwcHwcfOp::create(
                        builder, resultTy, ValueRange{input, filter},
                        op.getOutputs(), strides, dilations)
                        .getResult(0);

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
      SmallVector<ReassociationExprs> reassociationMap(3);
      for (int i = 0; i < 3; i++) {
        reassociationMap[i].push_back(builder.getAffineDimExpr(i));
      }
      reassociationMap.back().push_back(builder.getAffineDimExpr(3));

      auto expandTy =
          RankedTensorType::get({inputTy.getDimSize(0), inputTy.getDimSize(1),
                                 inputTy.getDimSize(2), 1},
                                accETy);
      inputSum = tensor::ExpandShapeOp::create(builder, expandTy, inputSum,
                                               reassociationMap);

      llvm::SmallVector<int64_t> poolDims;
      llvm::SmallVector<Value> poolDynDims;
      GetDynamicDym(builder, poolDims, poolDynDims, inputSum, 0);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, 1);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, 2);
      GetDynamicDym(builder, poolDims, poolDynDims, inputSum, 3);

      // Perform a sum-pooling operation across the kernel width and height.
      auto poolTy = RankedTensorType::get(poolDims, accETy);
      Value poolTensor = emptyZero(builder, poolTy, poolDynDims);

      // Create the empty kernel defining the shape for the pooling operation.
      llvm::SmallVector<int64_t> kDims;
      llvm::SmallVector<Value> kDyn;
      GetDynamicDym(builder, kDims, kDyn, filter, 0);
      GetDynamicDym(builder, kDims, kDyn, filter, 1);
      Value poolInit = tensor::EmptyOp::create(builder, kDims, accETy, kDyn);

      inputSum =
          linalg::PoolingNhwcSumOp::create(builder, ArrayRef<Type>{poolTy},
                                           ValueRange{inputSum, poolInit},
                                           poolTensor, strides, dilations)
              .getResult(0);

      // Collapse the length-1 ending dimension away.
      auto collapseTy =
          RankedTensorType::get(poolTy.getShape().drop_back(), accETy);
      inputSum = tensor::CollapseShapeOp::create(builder, collapseTy, inputSum,
                                                 reassociationMap);

      // Apply the zero-point update based on the input sum.
      newConv = applyZeroPoint(builder, newConv, inputSum, fZp, {0, 1, 2});
    }

    // Apply the final update that occurs when there are multiple zero-points.
    if (!iZpIsZero && !fZpIsZero) {
      Value count = multiplyDims(builder, filter, {0, 1, 2});
      Value cast = arith::IndexCastOp::create(builder, accETy, count);
      Value ifZp = arith::MulIOp::create(builder, iZp, fZp);
      Value zpUpdate = arith::MulIOp::create(builder, ifZp, cast);

      newConv = addScalar(builder, newConv, zpUpdate);
    }

    rewriter.replaceOp(op, newConv);
    return success();
  }
};

// Pattern lowering depthwise_conv_2d_nhwc_hwc_q to depthwise_conv_2d_nhwc_hwc.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
struct QuantizedDepthwiseConvToDepthwiseConv
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcQOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcQOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto input = op.getInputs()[0];
    auto filter = op.getInputs()[1];
    auto iZp = op.getInputs()[2];
    auto fZp = op.getInputs()[3];
    auto resultTy = cast<ShapedType>(op.getType(0));
    auto accETy = resultTy.getElementType();

    auto strides = op.getStrides();
    auto dilations = op.getDilations();

    IntegerAttr iZpConst, fZpConst;
    bool iZpIsZero = matchPattern(iZp, m_Constant(&iZpConst)) &&
                     iZpConst.getValue().isZero();
    bool fZpIsZero = matchPattern(fZp, m_Constant(&fZpConst)) &&
                     fZpConst.getValue().isZero();

    // First implement the convolution without the zero point.
    Value newConv = linalg::DepthwiseConv2DNhwcHwcOp::create(
                        builder, resultTy, ValueRange{input, filter},
                        op.getOutputs(), strides, dilations)
                        .getResult(0);

    // If the zero pointer value is zero we can just replace.
    if (iZpIsZero && fZpIsZero) {
      rewriter.replaceOp(op, newConv);
      return success();
    }

    // Compute the summation and correction for the filter zero point:
    //  sum(c) = filter(a, b, c)
    //  conv(a, b, c, d) -= iZp * sum(d)
    if (!iZpIsZero) {
      Value filterSum =
          sumReduceDimensionSubset(builder, filter, accETy,
                                   /*reduce_dim=*/{true, true, false});
      newConv = applyZeroPoint(builder, newConv, filterSum, iZp, {3});
    }

    if (!fZpIsZero) {
      llvm::SmallVector<int64_t> poolDims;
      llvm::SmallVector<Value> poolDynDims;
      GetDynamicDym(builder, poolDims, poolDynDims, input, 0);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, 1);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, 2);
      GetDynamicDym(builder, poolDims, poolDynDims, input, 3);

      // Perform a sum-pooling operation across the kernel width and height.
      auto poolTy = RankedTensorType::get(poolDims, accETy);
      Value poolTensor = emptyZero(builder, poolTy, poolDynDims);

      // Create the empty kernel defining the shape for the pooling operation.
      llvm::SmallVector<int64_t> kDims;
      llvm::SmallVector<Value> kDyn;
      GetDynamicDym(builder, kDims, kDyn, filter, 0);
      GetDynamicDym(builder, kDims, kDyn, filter, 1);
      Value poolInit = tensor::EmptyOp::create(builder, kDims, accETy, kDyn);

      Value inputSum =
          linalg::PoolingNhwcSumOp::create(builder, ArrayRef<Type>{poolTy},
                                           ValueRange{input, poolInit},
                                           poolTensor, strides, dilations)
              .getResult(0);

      // Apply the zero-point update based on the input sum.
      newConv = applyZeroPoint(builder, newConv, inputSum, fZp, {0, 1, 2, 3});
    }

    // Apply the final update that occurs when there are multiple zero-points.
    if (!iZpIsZero && !fZpIsZero) {
      Value count = multiplyDims(builder, filter, {0, 1});
      Value cast = arith::IndexCastOp::create(builder, accETy, count);

      Value ifZp = arith::MulIOp::create(builder, iZp, fZp);
      Value zpUpdate = arith::MulIOp::create(builder, ifZp, cast);

      newConv = addScalar(builder, newConv, zpUpdate);
    }

    rewriter.replaceOp(op, newConv);
    return success();
  }
};

/// Pass that lowers quantized_conv to conv.
class LinalgQuantizedConvToConvPass final
    : public impl::LinalgQuantizedConvToConvPassBase<
          LinalgQuantizedConvToConvPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    linalg::populateSimplifyDepthwiseConvPatterns(patterns);
    patterns.add<QuantizedConvToConv, QuantizedDepthwiseConvToDepthwiseConv>(
        context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
