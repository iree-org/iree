// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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
  return rewriter.create<tensor::EmptyOp>(mixedSizes, eTy);
}

// Creates an zero initialized tensor of given shape and type.
Value emptyZero(ImplicitLocOpBuilder &builder, RankedTensorType ty,
                llvm::SmallVector<Value> dyn) {
  Value empty =
      builder.create<tensor::EmptyOp>(ty.getShape(), ty.getElementType(), dyn);

  TypedAttr attr = builder.getZeroAttr(ty.getElementType());
  Value cnst = builder.create<arith::ConstantOp>(attr);
  return builder.create<linalg::FillOp>(ValueRange{cnst}, ValueRange{empty})
      .result();
}

// Apply the multiply subtract corresponding with a zero-point adjustment
// broadcasting according to the afifne map.
Value applyZeroPoint(ImplicitLocOpBuilder &builder, Value conv, Value sum,
                     Value zp, ArrayRef<int> affine_map) {
  auto context = builder.getContext();
  auto convTy = llvm::cast<RankedTensorType>(conv.getType());

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
  return builder
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

// Add the scalar value to the tensor.
Value addScalar(ImplicitLocOpBuilder &builder, Value value, Value scalar) {
  auto ty = llvm::cast<RankedTensorType>(value.getType());
  SmallVector<utils::IteratorType> iterators(ty.getRank(),
                                             utils::IteratorType::parallel);
  auto map = builder.getMultiDimIdentityMap(ty.getRank());
  Value init = emptyCopy(builder, value);
  return builder
      .create<linalg::GenericOp>(
          init.getType(), ValueRange{value}, ValueRange{init},
          ArrayRef<AffineMap>{map, map}, iterators,
          [=](OpBuilder &b, Location loc, ValueRange args) {
            Value add = b.create<arith::AddIOp>(loc, args[0], scalar);
            b.create<linalg::YieldOp>(loc, add);
          })
      .getResult(0);
}

void GetDynamicDym(ImplicitLocOpBuilder &builder,
                   llvm::SmallVector<int64_t> &dims,
                   llvm::SmallVector<Value> &dynDims, Value value,
                   int64_t dim) {
  ShapedType ty = llvm::cast<ShapedType>(value.getType());
  dims.push_back(ty.getDimSize(dim));
  if (ty && ty.isDynamicDim(dim))
    dynDims.push_back(builder.create<tensor::DimOp>(value, dim));
}

Value multiplyDims(ImplicitLocOpBuilder &builder, Value value,
                   llvm::ArrayRef<int64_t> dims) {
  Value count = builder.create<tensor::DimOp>(value, dims.front());

  for (auto d : dims.drop_front()) {
    Value dim = builder.create<tensor::DimOp>(value, d);
    count = builder.create<arith::MulIOp>(count, dim);
  }

  return count;
}

// Pattern lowering conv_2d_nhwc_hwcf_q to conv_2d_nhwc_hwcf.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
template <typename Conv2DQOpType, typename Conv2DOpType,
          typename ConvPoolingType>
struct QuantizedConvToConv : public OpRewritePattern<Conv2DQOpType> {
  using OpRewritePattern<Conv2DQOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DQOpType op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto input = op.getInputs()[0];
    auto filter = op.getInputs()[1];
    auto iZp = op.getInputs()[2];
    auto fZp = op.getInputs()[3];
    auto resultTy = llvm::cast<ShapedType>(op.getType(0));
    auto accETy = resultTy.getElementType();
    auto strides = op.getStrides();
    auto dilations = op.getDilations();

    bool isNHWC = isConvNHWC(op);
    int filterFDimIndex, filterHDimIndex, filterWDimIndex, filterCDimIndex;
    int inputNDimIndex, inputHDimIndex, inputWDimIndex, inputCDimIndex;
    getConv2DOpIndices(isNHWC, filterFDimIndex, filterCDimIndex,
                       filterHDimIndex, filterWDimIndex, inputNDimIndex,
                       inputHDimIndex, inputWDimIndex, inputCDimIndex);

    IntegerAttr iZpConst, fZpConst;
    bool iZpIsZero = matchPattern(iZp, m_Constant(&iZpConst)) &&
                     iZpConst.getValue().isZero();
    bool fZpIsZero = matchPattern(fZp, m_Constant(&fZpConst)) &&
                     fZpConst.getValue().isZero();

    // First implement the convolution without the zero point.
    Value newConv =
        builder
            .create<Conv2DOpType>(resultTy, ValueRange{input, filter},
                                  op.getOutputs(), strides, dilations)
            .getResult(0);

    // If the zero pointer value is zero we can just replace.
    if (iZpIsZero && fZpIsZero) {
      rewriter.replaceOp(op, newConv);
      return success();
    }

    // Compute the summation and correction for the filter zero point:
    //  filter(h, w, c) = reduce(filter(h, w, c, f), f)
    //  conv(n, h, w, c) -= iZp * filter(h, w, c)
    if (!iZpIsZero) {
      newConv =
          applyFilterZeroPointCorrection(builder, newConv, filter, iZp, accETy,
                                         filterFDimIndex, inputCDimIndex);
    }

    if (!fZpIsZero) {
      newConv = applyInputZeroPointCorrection(
          isNHWC, builder, newConv, input, filter, fZp, accETy, strides,
          dilations, inputNDimIndex, inputHDimIndex, inputWDimIndex,
          inputCDimIndex, filterHDimIndex, filterWDimIndex);
    }

    // Apply the final update that occurs when there are multiple zero-points.
    if (!iZpIsZero && !fZpIsZero) {
      newConv = applyFinalZeroPointUpdate(builder, newConv, filter, iZp, fZp,
                                          accETy, filterHDimIndex,
                                          filterWDimIndex, filterCDimIndex);
    }

    rewriter.replaceOp(op, newConv);
    return success();
  }

private:
  bool isConvNHWC(Conv2DQOpType op) const {
    if (auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfQOp>(&op)) {
      return true;
    } else if (auto convOp = dyn_cast<linalg::Conv2DNchwFchwQOp>(&op)) {
      return false;
    } else {
      llvm_unreachable("Unsupported Conv2DOp type!");
    }
  }

  void getConv2DOpIndices(bool isNHWC, int &filterFDimIndex,
                          int &filterCDimIndex, int &filterHDimIndex,
                          int &filterWDimIndex, int &inputNDimIndex,
                          int &inputHDimIndex, int &inputWDimIndex,
                          int &inputCDimIndex) const {
    if (isNHWC) {
      // hwcf
      filterHDimIndex = 0;
      filterWDimIndex = 1;
      filterCDimIndex = 2;
      filterFDimIndex = 3;
      // nhwc
      inputNDimIndex = 0;
      inputHDimIndex = 1;
      inputWDimIndex = 2;
      inputCDimIndex = 3;
    } else {
      // fchw
      filterFDimIndex = 0;
      filterCDimIndex = 1;
      filterHDimIndex = 2;
      filterWDimIndex = 3;
      // nchw
      inputNDimIndex = 0;
      inputCDimIndex = 1;
      inputHDimIndex = 2;
      inputWDimIndex = 3;
    }
  }

  Value applyFilterZeroPointCorrection(ImplicitLocOpBuilder &builder,
                                       Value newConv, Value filter, Value iZp,
                                       Type accETy, int filterFDimIndex,
                                       int inputCDimIndex) const {
    llvm::SmallVector<bool, 4> reduceDim(4, true);
    reduceDim[filterFDimIndex] = false;
    Value filterSum =
        sumReduceDimensionSubset(builder, filter, accETy, reduceDim);
    return applyZeroPoint(builder, newConv, filterSum, iZp, {inputCDimIndex});
  }

  Value applyInputZeroPointCorrection(
      bool isNHWC, ImplicitLocOpBuilder &builder, Value newConv, Value input,
      Value filter, Value fZp, Type accETy, DenseIntElementsAttr strides,
      DenseIntElementsAttr dilations, int inputNDimIndex, int inputHDimIndex,
      int inputWDimIndex, int inputCDimIndex, int filterHDimIndex,
      int filterWDimIndex) const {
    // Reduce along the input feature dimension:
    //   sum(n, h, w) = input(n, h, w, c)
    llvm::SmallVector<bool, 4> reduceDim(4, false);
    reduceDim[inputCDimIndex] = true;
    Value inputSum =
        sumReduceDimensionSubset(builder, input, accETy, reduceDim);
    inputSum = expandAndPoolInputSum(
        isNHWC, builder, input, inputSum, newConv, filter, accETy, strides,
        dilations, inputNDimIndex, inputHDimIndex, inputWDimIndex,
        inputCDimIndex, filterHDimIndex, filterWDimIndex);
    // Apply the zero-point update based on the input sum.
    return applyZeroPoint(builder, newConv, inputSum, fZp,
                          {inputNDimIndex, inputHDimIndex, inputWDimIndex});
  }

  Value expandAndPoolInputSum(bool isNHWC, ImplicitLocOpBuilder &builder,
                              Value input, Value inputSum, Value newConv,
                              Value filter, Type accETy,
                              DenseIntElementsAttr strides,
                              DenseIntElementsAttr dilations,
                              int inputNDimIndex, int inputHDimIndex,
                              int inputWDimIndex, int inputCDimIndex,
                              int filterHDimIndex, int filterWDimIndex) const {
    auto reassociationMap =
        getReassociationMap(isNHWC, builder, inputNDimIndex, inputHDimIndex,
                            inputWDimIndex, inputCDimIndex);
    auto expandTy = getExpandedType(input, accETy, inputCDimIndex);
    inputSum = builder.create<tensor::ExpandShapeOp>(expandTy, inputSum,
                                                     reassociationMap);

    llvm::SmallVector<int64_t> poolDims;
    llvm::SmallVector<Value> poolDynDims;
    if (isNHWC) {
      GetDynamicDym(builder, poolDims, poolDynDims, inputSum, inputNDimIndex);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, inputHDimIndex);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, inputWDimIndex);
      GetDynamicDym(builder, poolDims, poolDynDims, inputSum, inputCDimIndex);
    } else {
      GetDynamicDym(builder, poolDims, poolDynDims, inputSum, inputNDimIndex);
      GetDynamicDym(builder, poolDims, poolDynDims, inputSum, inputCDimIndex);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, inputHDimIndex);
      GetDynamicDym(builder, poolDims, poolDynDims, newConv, inputWDimIndex);
    }

    // Perform a sum-pooling operation across the kernel width and height.
    auto poolTy = RankedTensorType::get(poolDims, accETy);
    Value poolTensor = emptyZero(builder, poolTy, poolDynDims);

    llvm::SmallVector<int64_t> kDims;
    llvm::SmallVector<Value> kDyn;
    GetDynamicDym(builder, kDims, kDyn, filter, filterHDimIndex);
    GetDynamicDym(builder, kDims, kDyn, filter, filterWDimIndex);
    Value poolInit = builder.create<tensor::EmptyOp>(kDims, accETy, kDyn);

    inputSum = builder
                   .create<ConvPoolingType>(ArrayRef<Type>{poolTy},
                                            ValueRange{inputSum, poolInit},
                                            poolTensor, strides, dilations)
                   .getResult(0);

    // Collapse the length-1 ending dimension away.
    ArrayRef<int64_t> collapseShape;
    llvm::SmallVector<int64_t, 4> new_shape;
    if (isNHWC) {
      collapseShape = poolTy.getShape().drop_back();
    } else {
      auto shape = poolTy.getShape();
      new_shape = {shape[0], shape[2], shape[3]};
      collapseShape = new_shape;
    }

    auto collapseTy = RankedTensorType::get(collapseShape, accETy);

    return builder.create<tensor::CollapseShapeOp>(collapseTy, inputSum,
                                                   reassociationMap);
  }

  Value applyFinalZeroPointUpdate(ImplicitLocOpBuilder &builder, Value newConv,
                                  Value filter, Value iZp, Value fZp,
                                  Type accETy, int filterHDimIndex,
                                  int filterWDimIndex,
                                  int filterCDimIndex) const {
    Value count = multiplyDims(
        builder, filter, {filterHDimIndex, filterWDimIndex, filterCDimIndex});
    Value cast = builder.create<arith::IndexCastOp>(accETy, count);
    Value ifZp = builder.create<arith::MulIOp>(iZp, fZp);
    Value zpUpdate = builder.create<arith::MulIOp>(ifZp, cast);
    return addScalar(builder, newConv, zpUpdate);
  }

  SmallVector<ReassociationExprs>
  getReassociationMap(bool isNHWC, ImplicitLocOpBuilder &builder,
                      int inputNDimIndex, int inputHDimIndex,
                      int inputWDimIndex, int inputCDimIndex) const {
    // Materialize a length-1 dimension at the C dimension of the input.
    SmallVector<ReassociationExprs> reassociationMap(3);
    if (isNHWC) {
      for (int i = 0; i < 3; i++)
        reassociationMap[i].push_back(builder.getAffineDimExpr(i));
      reassociationMap.back().push_back(builder.getAffineDimExpr(3));
    } else {
      reassociationMap[0].push_back(builder.getAffineDimExpr(0));
      reassociationMap[0].push_back(builder.getAffineDimExpr(1));
      reassociationMap[1].push_back(builder.getAffineDimExpr(2));
      reassociationMap[2].push_back(builder.getAffineDimExpr(3));
    }
    return reassociationMap;
  }

  RankedTensorType getExpandedType(Value input, Type accETy,
                                   int inputCDimIndex) const {
    auto type = cast<RankedTensorType>(input.getType());
    auto shape = type.getShape();
    llvm::SmallVector<int64_t, 4> new_shape(shape.begin(), shape.end());
    new_shape[inputCDimIndex] = 1;
    return RankedTensorType::get(new_shape, accETy);
  }
};

// Pattern lowering depthwise_conv_2d_nhwc_hwc_q to depthwise_conv_2d_nhwc_hwc.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
struct QuantizedDepthwiseConvToDepthwiseConv
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcQOp> {
  using OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcQOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto input = op.getInputs()[0];
    auto filter = op.getInputs()[1];
    auto iZp = op.getInputs()[2];
    auto fZp = op.getInputs()[3];
    auto resultTy = llvm::cast<ShapedType>(op.getType(0));
    auto accETy = resultTy.getElementType();

    auto strides = op.getStrides();
    auto dilations = op.getDilations();

    IntegerAttr iZpConst, fZpConst;
    bool iZpIsZero = matchPattern(iZp, m_Constant(&iZpConst)) &&
                     iZpConst.getValue().isZero();
    bool fZpIsZero = matchPattern(fZp, m_Constant(&fZpConst)) &&
                     fZpConst.getValue().isZero();

    // First implement the convolution without the zero point.
    Value newConv = builder
                        .create<linalg::DepthwiseConv2DNhwcHwcOp>(
                            resultTy, ValueRange{input, filter},
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
      Value poolInit = builder.create<tensor::EmptyOp>(kDims, accETy, kDyn);

      Value inputSum =
          builder
              .create<linalg::PoolingNhwcSumOp>(ArrayRef<Type>{poolTy},
                                                ValueRange{input, poolInit},
                                                poolTensor, strides, dilations)
              .getResult(0);

      // Apply the zero-point update based on the input sum.
      newConv = applyZeroPoint(builder, newConv, inputSum, fZp, {0, 1, 2, 3});
    }

    // Apply the final update that occurs when there are multiple zero-points.
    if (!iZpIsZero && !fZpIsZero) {
      Value count = multiplyDims(builder, filter, {0, 1});
      Value cast = builder.create<arith::IndexCastOp>(accETy, count);

      Value ifZp = builder.create<arith::MulIOp>(iZp, fZp);
      Value zpUpdate = builder.create<arith::MulIOp>(ifZp, cast);

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
    linalg::populateLinalgNamedOpConversionPatterns(patterns);
    patterns.add<
        QuantizedConvToConv<linalg::Conv2DNhwcHwcfQOp, linalg::Conv2DNhwcHwcfOp,
                            linalg::PoolingNhwcSumOp>,
        QuantizedConvToConv<linalg::Conv2DNchwFchwQOp, linalg::Conv2DNchwFchwOp,
                            linalg::PoolingNchwSumOp>,
        QuantizedDepthwiseConvToDepthwiseConv>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
