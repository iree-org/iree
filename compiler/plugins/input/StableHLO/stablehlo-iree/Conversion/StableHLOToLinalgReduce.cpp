// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO reduction ops to Linalg dialect.
// These patterns are separated out to their own file to save on the compilation
// times.

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/Conversion/LegalizeToLinalgUtils.h"
#include "stablehlo-iree/Conversion/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
/// Returns true when reduction `op` is not supported and should be filtered
/// out.
static bool isUnsupported(mlir::stablehlo::ReduceOp op) {
  // Empty reductions are not supported. We expect canonicalization patterns to
  // handle them.
  if (op.getDimensions().empty())
    return true;

  // We require all reduce shapes to be the same, up to the element types, so
  // we can just the first operand and the first result as a representative.
  if (auto inputTy =
          dyn_cast<RankedTensorType>(op.getInputs().getType().front())) {
    return llvm::is_contained(inputTy.getShape(), 0);
  }

  return false;
}

/// Returns a permutation AffineMap that puts all reduction dimensions to the
/// last. The order of parallel loops and reduction loops are all sorted. E.g.,
/// if `rank` is 4 and `reductionDims` is {1, 3}, then
/// "(d0, d1, d2, d3) -> (d0, d2, d1, d3)" is used. The inverse permutation of
/// the AffineMap is returned.
AffineMap getTransposeMapForReduction(MLIRContext *context, int rank,
                                      ArrayRef<int64_t> reductionDims) {
  llvm::SmallSetVector<int, 4> s(reductionDims.begin(), reductionDims.end());

  SmallVector<unsigned> permutation;
  for (int i = 0; i < rank; ++i) {
    if (!s.contains(i)) {
      permutation.push_back(i);
    }
  }

  llvm::append_range(permutation, reductionDims);
  auto map = AffineMap::getPermutationMap(permutation, context);
  return inversePermutation(map);
}

SmallVector<Value, 8>
getReduceOpEmptyTensorDynSizes(OpBuilder &b, Location loc, Value arg,
                               ShapedType resultType,
                               ArrayRef<int64_t> reductionDims) {
  llvm::SmallSetVector<int, 4> s(reductionDims.begin(), reductionDims.end());

  SmallVector<unsigned> parallelDims;
  SmallVector<Value, 8> dynShape;
  int rank = cast<RankedTensorType>(arg.getType()).getRank();
  for (int i = 0, j = 0; i < rank; ++i) {
    if (s.contains(i))
      continue;
    if (!resultType.isDynamicDim(j++))
      continue;
    dynShape.push_back(b.create<tensor::DimOp>(loc, arg, i));
  }

  return dynShape;
}

struct ReduceRegionReturnOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isInBodyOfLinalgOps(op)) {
      return failure();
    }

    SmallVector<Value> operands(adaptor.getOperands());
    for (Value &operand : operands) {
      if (isa<ShapedType>(operand.getType())) {
        Location loc = operand.getLoc();
        operand = rewriter.create<tensor::ExtractOp>(loc, operand);
      }
    }
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, operands);
    return success();
  }
};

struct ReduceOpToGenericConverter final
    : OpConversionPattern<mlir::stablehlo::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isUnsupported(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported reduce (noop or empty)");
    }

    Location loc = op.getLoc();

    int numOperands = static_cast<int>(adaptor.getInputs().size());

    if (llvm::any_of(adaptor.getInputs(), [](Value v) {
          return !isa<RankedTensorType>(v.getType());
        })) {
      return rewriter.notifyMatchFailure(op, "expects known-rank args");
    }
    auto srcRank = cast<ShapedType>(adaptor.getInputs()[0].getType()).getRank();

    SmallVector<int64_t> reductionDims = extract1DVector(op.getDimensions());

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    SmallVector<Value> outputs;
    SmallVector<AffineMap, 3> indexingMaps;
    for (auto [operand, initValue, resultType] : llvm::zip_equal(
             adaptor.getInputs(), adaptor.getInitValues(), resultTypes)) {
      // Check if init_value is constant. If so, inline the value into the
      // region.
      initValue = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);

      SmallVector<Value, 8> dynShape = getReduceOpEmptyTensorDynSizes(
          rewriter, loc, operand, cast<ShapedType>(resultType), reductionDims);
      auto emptyTensor =
          getEmptyTensor(rewriter, loc, cast<ShapedType>(resultType), dynShape);
      Value filledTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor).result();
      outputs.push_back(filledTensor);
    }

    // Prepare indexing maps for linalg generic op. The elements are for src
    // and dst. Transpose `src` to make the reduction loops be the innermost,
    // because it's easier to fully utilize processors.
    indexingMaps.append(numOperands,
                        getTransposeMapForReduction(rewriter.getContext(),
                                                    static_cast<int>(srcRank),
                                                    reductionDims));

    // The indexing map of `dst` should drop the reduction loops. Since the
    // reduction loops now are all in the innermost, drops
    // `reduction_dims.size()` dimensions. We don't need an inverse
    // permutation here because they are the same.
    SmallVector<AffineExpr> exprs;
    for (int i = 0, e = srcRank - reductionDims.size(); i < e; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    indexingMaps.append(numOperands,
                        AffineMap::get(srcRank, /*symbolCount=*/0, exprs,
                                       rewriter.getContext()));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/resultTypes, adaptor.getInputs(),
        /*outputBuffers=*/ValueRange{outputs}, indexingMaps,
        getParallelAndReductionIterators(srcRank, reductionDims.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Convert the signature of the body. The reduce op region apply function
    // has a signature (lhs, rhs) -> output, all of the same tensor type t.
    // This is converted to a function with the same signature but with
    // element types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will
    // be converted to "(f32, f32, f32)".
    Region &region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getBody(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(numOperands * 2);

    // The stablehlo ReduceOp requires that the seed be used as a LHS operand
    // inside the region, and the seed is encoded in linalg in the initial out
    // value, so modify the signature of the block and the value mappings, so
    // the output args will correlate with the original LHS and the inputs
    // correlate with the original RHS.
    for (auto [idx, val] : llvm::enumerate(op.getInputs())) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx + numOperands,
          // type for the new operand number 'idx'.
          typeConverter->convertType(
              cast<ShapedType>(val.getType()).getElementType()));
    }
    for (auto [idx, val] : llvm::enumerate(op.getInitValues())) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx,
          // type for the new operand number 'idx' + 'numOperands'.
          typeConverter->convertType(
              cast<ShapedType>(val.getType()).getElementType()));
    }

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

struct ReduceOpToReduceConverter final
    : OpConversionPattern<mlir::stablehlo::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isUnsupported(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported reduce (noop or empty)");
    }

    auto reductionDims =
        llvm::to_vector(op.getDimensions().getValues<int64_t>());
    // stablehlo.reduce doesn't specify the order of the reduction dimensions.
    llvm::sort(reductionDims);

    auto toRankedTensor = [](Value v) -> RankedTensorType {
      return dyn_cast<RankedTensorType>(v.getType());
    };

    SmallVector<Value> outputs;
    SmallVector<RankedTensorType> operandTypes, initTypes;
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    Location loc = op.getLoc();
    for (auto [operand, initValue, resultType] : llvm::zip_equal(
             adaptor.getInputs(), adaptor.getInitValues(), resultTypes)) {
      auto initType = toRankedTensor(initValue);
      if (!initType)
        return rewriter.notifyMatchFailure(op,
                                           "expects known-rank init values");
      initTypes.push_back(initType);
      auto operandType = toRankedTensor(operand);
      if (!operandType)
        return rewriter.notifyMatchFailure(op, "expects known-rank operands");
      operandTypes.push_back(operandType);
      initValue = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
      auto tensorResultType = cast<RankedTensorType>(resultType);
      // For linalg.reduce, the result type's dimensions must match the input's
      // dimensions, whereas StableHLO allows replacing static dimensions with
      // dynamic ones.
      SmallVector<int64_t> resultShape;
      SmallVector<Value, 8> dynShape;
      for (auto [index, dim] :
           llvm::enumerate(cast<ShapedType>(operand.getType()).getShape())) {
        if (!llvm::is_contained(reductionDims, index)) {
          resultShape.push_back(dim);
          if (ShapedType::isDynamic(dim)) {
            dynShape.push_back(
                rewriter.create<tensor::DimOp>(loc, operand, index));
          }
        }
      }

      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultShape, tensorResultType.getElementType(), dynShape);
      Value filledTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor).result();
      outputs.push_back(filledTensor);
    }

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getInputs(), outputs, reductionDims,
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    Region &region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getBody(), region, region.end());

    // Convert the signature of the body. The reduce op 'computation' region
    // apply function has a signature with tensor types, this is converted to a
    // function with element types. E.g. the signature "(tensor<f32>,
    // tensor<f32>) -> tensor<f32>" will be converted to "(f32, f32) -> f32".
    // Also, we need to swap the operands of the function. The stablehlo.reduce
    // op expects the init values to be the first parameters of the apply
    // function, while the linalg.reduction op expects the init values as the
    // last parameters of the 'combiner' region apply function.
    TypeConverter::SignatureConversion signatureConverter(
        linalgOp.getNumDpsInputs() * 2);
    assert(linalgOp.getNumDpsInputs() == linalgOp.getNumDpsInits());
    for (const auto &[idx, val] : llvm::enumerate(operandTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx + linalgOp.getNumDpsInputs(),
          // type for new operand number 'idx'.
          typeConverter->convertType(val.getElementType()));
    }
    for (const auto &[idx, val] : llvm::enumerate(initTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx,
          // type for new operand number 'idx' + linalgOp.getNumInputs()
          typeConverter->convertType(val.getElementType()));
    }
    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());

    // Cast the result to the correct type.
    SmallVector<Value> results;
    for (auto [result, resultType] :
         llvm::zip(linalgOp.getResults(), resultTypes)) {
      results.push_back(
          rewriter.createOrFold<tensor::CastOp>(loc, resultType, result));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ReduceWindowOpOnTensorsGenericConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op->getContext();
    Location loc = op.getLoc();
    llvm::SmallVector<Value> initValues = adaptor.getInitValues();
    llvm::SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    auto numOperands = initValues.size();

    llvm::SmallVector<int64_t> windowDimensions =
        extract1DVector(op.getWindowDimensions());

    llvm::SmallVector<int64_t> padding;
    if (op.getPadding()) {
      padding = extract1DVector(*op.getPadding());
    }

    llvm::SmallVector<int64_t> baseDilations;
    if (op.getBaseDilations()) {
      baseDilations = extract1DVector(*op.getBaseDilations());
    }

    llvm::SmallVector<int64_t> windowStrides(windowDimensions.size(), 1);
    if (op.getWindowStrides()) {
      windowStrides = extract1DVector(*op.getWindowStrides());
    }

    llvm::SmallVector<int64_t> windowDilations(windowDimensions.size(), 1);
    if (op.getWindowDilations()) {
      windowDilations = extract1DVector(*op.getWindowDilations());
    }

    auto rank = static_cast<int64_t>(windowDimensions.size());
    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> windowExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    SmallVector<int64_t> filteredWindowDims;

    int windowDim = 0;
    for (int64_t i = 0; i < rank; i++) {
      AffineExpr srcExpr = mlir::getAffineDimExpr(i, ctx);

      if (windowStrides[i] != 1)
        srcExpr = srcExpr * windowStrides[i];

      if (windowDimensions[i] != 1) {
        filteredWindowDims.push_back(windowDimensions[i]);
        AffineExpr windowExpr = mlir::getAffineDimExpr(rank + windowDim, ctx);
        windowExprs.push_back(windowExpr);

        if (windowDilations[i] != 1)
          windowExpr = windowExpr * windowDilations[i];

        srcExpr = srcExpr + windowExpr;
        windowDim++;
      }

      srcExprs.push_back(srcExpr);
      dstExprs.push_back(mlir::getAffineDimExpr(i, ctx));
    }

    SmallVector<AffineMap> inferredMaps(3, AffineMap::get(ctx));
    if (rank > 0) {
      inferredMaps =
          AffineMap::inferFromExprList({srcExprs, windowExprs, dstExprs});
    }

    SmallVector<AffineMap> indexingMaps;

    indexingMaps.append(numOperands, inferredMaps[0]);
    indexingMaps.append(1, inferredMaps[1]);
    indexingMaps.append(numOperands, inferredMaps[2]);

    // Setup the initial values.
    llvm::SmallVector<Value> broadcastValues;
    for (uint64_t i = 0, s = initValues.size(); i < s; i++) {
      Value initValue = initValues[i];
      auto resultTy = llvm::cast<ShapedType>(resultTypes[i]);
      if (!resultTy.hasStaticShape())
        return failure();

      auto broadcastSizes = rewriter.getI64TensorAttr(resultTy.getShape());
      broadcastValues.push_back(rewriter.create<mlir::stablehlo::BroadcastOp>(
          loc, resultTy, initValue, broadcastSizes));
    }

    llvm::SmallVector<Value> inputs = llvm::to_vector(adaptor.getInputs());

    // Pad as necessary.
    if (llvm::any_of(padding, [](int64_t v) { return v != 0; }) ||
        llvm::any_of(baseDilations, [](int64_t v) { return v != 1; })) {
      llvm::SmallVector<int64_t> staticLows(rank, 0);
      llvm::SmallVector<int64_t> staticHighs(rank, 0);
      for (int64_t i = 0; i < static_cast<int64_t>(padding.size()); i += 2) {
        staticLows[i / 2] = padding[i];
        staticHighs[i / 2] = padding[i + 1];
      }
      // Translate base dilation into interior padding.
      llvm::SmallVector<int64_t> staticInteriors(rank, 0);
      for (auto [idx, dilation] : llvm::enumerate(baseDilations)) {
        staticInteriors[idx] = dilation - 1;
      }

      auto padAttrType =
          RankedTensorType::get({rank}, rewriter.getIntegerType(64));
      auto padLows = DenseIntElementsAttr::get(padAttrType, staticLows);
      auto padHighs = DenseIntElementsAttr::get(padAttrType, staticHighs);
      auto padInteriors =
          DenseIntElementsAttr::get(padAttrType, staticInteriors);

      for (auto [input, initValue] : llvm::zip(inputs, initValues)) {
        input = rewriter.create<mlir::stablehlo::PadOp>(
            loc, input, initValue, padLows, padHighs, padInteriors);
      }
    }

    // Add the extra input for the reduction dimension.
    inputs.push_back(rewriter.create<tensor::EmptyOp>(loc, filteredWindowDims,
                                                      rewriter.getF32Type()));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/resultTypes,
        /*inputs=*/inputs,
        /*outputs=*/broadcastValues, indexingMaps,
        getParallelAndReductionIterators(rank + filteredWindowDims.size(),
                                         filteredWindowDims.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Convert the signature of the body. This includes converting scalar
    // tensors to their scalar values and inserting an additional block arg for
    // the window arg.
    Region &region = linalgOp.getRegion();
    rewriter.cloneRegionBefore(op.getBody(), region, region.end());

    TypeConverter::SignatureConversion signatureConverter(
        inputs.size() + op->getNumResults() - 1);

    // ReduceWindow requires that the seed be used as a LHS operand inside the
    // region, and the seed is encoded in linalg in the initial out value, so
    // modify the signature of the block and the value mappings, so the output
    // args will correlate with the LHS and the inputs correlate with the RHS.
    for (auto [i, type] : llvm::enumerate(resultTypes)) {
      auto idx = inputs.size() + i - 1;
      signatureConverter.addInputs(idx,
                                   cast<ShapedType>(type).getElementType());
    }

    signatureConverter.addInputs(
        cast<ShapedType>(inputs.back().getType()).getElementType());

    for (auto [i, input] :
         llvm::enumerate(ArrayRef<Value>(inputs).drop_back())) {
      signatureConverter.addInputs(
          i, cast<ShapedType>(input.getType()).getElementType());
    }

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

struct ReduceWindowOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern::OpConversionPattern;

  /// Get the operation used for reduction applied to `result_index`th result.
  /// Its expected to be a binary operation that consumes `result_index`th and
  /// `result_index + getInputs().size`th arguments of the body.
  static Operation *getReductionOp(mlir::stablehlo::ReduceWindowOp op,
                                   int resultIndex) {
    auto returnOp =
        cast<mlir::stablehlo::ReturnOp>(op.getBody().front().getTerminator());
    Operation *computeOp = returnOp.getResults()[resultIndex].getDefiningOp();
    if (computeOp->getNumOperands() != 2)
      return nullptr;
    auto arg0 = llvm::dyn_cast<BlockArgument>(computeOp->getOperand(0));
    auto arg1 = llvm::dyn_cast<BlockArgument>(computeOp->getOperand(1));
    if (!arg0 || !arg1)
      return nullptr;
    int64_t arg0Num = arg0.getArgNumber();
    int64_t arg1Num = arg1.getArgNumber();
    int64_t otherArgIndex = resultIndex + op.getInputs().size();
    if (arg0Num == resultIndex && arg1Num == otherArgIndex)
      return computeOp;
    if (arg0Num == otherArgIndex && arg1Num == resultIndex &&
        computeOp->hasTrait<mlir::OpTrait::IsCommutative>())
      return computeOp;
    return nullptr;
  }

  /// stablehlo.reduce_window is mapped to a linalg.pooling operation. The type
  /// of the pooling is determined based on the body of the reduce window
  /// operation. This class enumerates the different variants.
  enum class PoolingType {
    kInvalid,
    k2DMin,
    k3DMin,
    k2DMax,
    k3DMax,
    k2DAdd,
    k3DAdd,
  };

  static PoolingType getPoolingType(mlir::stablehlo::ReduceWindowOp reduceOp,
                                    int resultIndex) {
    auto rank = llvm::cast<ShapedType>(reduceOp.getResultTypes()[resultIndex])
                    .getRank();
    if (Operation *op = getReductionOp(reduceOp, resultIndex)) {
      if (isa<mlir::stablehlo::MinOp>(*op) && rank == 4)
        return PoolingType::k2DMin;
      if (isa<mlir::stablehlo::MinOp>(*op) && rank == 5)
        return PoolingType::k3DMin;
      if (isa<mlir::stablehlo::MaxOp>(*op) && rank == 4)
        return PoolingType::k2DMax;
      if (isa<mlir::stablehlo::MaxOp>(*op) && rank == 5)
        return PoolingType::k3DMax;
      if (isa<mlir::stablehlo::AddOp>(*op) && rank == 4)
        return PoolingType::k2DAdd;
      if (isa<mlir::stablehlo::AddOp>(*op) && rank == 5)
        return PoolingType::k3DAdd;
    }
    return PoolingType::kInvalid;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int rank = llvm::cast<ShapedType>(op.getResultTypes()[0]).getRank();
    if (rank != 4 && rank != 5) {
      return rewriter.notifyMatchFailure(
          op, "expected NHWC/NDHWC pooling-based op");
    }

    if (op.getPadding() && !isSplatValue(*op.getPadding(), 0)) {
      return rewriter.notifyMatchFailure(op, "require paddings are all zero");
    }

    if (op.getBaseDilations() && !isSplatValue(*op.getBaseDilations(), 1)) {
      return rewriter.notifyMatchFailure(op, "expected undilated base");
    }

    int lastDim = rank - 1;
    SmallVector<int64_t, 2> fakeWindowShapes;
    for (int i = 1; i < lastDim; ++i) {
      fakeWindowShapes.push_back(
          op.getWindowDimensions().getValues<int64_t>()[i]);
    }

    if (op.getWindowStrides() &&
        (op.getWindowStrides().value().getValues<int64_t>()[0] != 1 ||
         op.getWindowStrides().value().getValues<int64_t>()[lastDim] != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_strides to be [1,x,y,(z),1]");
    }
    if (op.getWindowDimensions() &&
        (op.getWindowDimensions().getValues<int64_t>()[0] != 1 ||
         op.getWindowDimensions().getValues<int64_t>()[lastDim] != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_dimensions to be [1,x,y,(z),1]");
    }

    Attribute strides;
    SmallVector<int64_t> vec;
    if (op.getWindowStridesAttr()) {
      for (int i = 1; i < lastDim; ++i) {
        vec.push_back(op.getWindowStrides().value().getValues<int64_t>()[i]);
      }
    } else {
      vec.assign(rank - 2, 1);
    }
    strides = rewriter.getI64VectorAttr(vec);

    Attribute dilations;
    vec.clear();
    if (op.getWindowDilations()) {
      for (int i = 1; i < lastDim; ++i) {
        vec.push_back(op.getWindowDilations().value().getValues<int64_t>()[i]);
      }
    } else {
      vec.assign(rank - 2, 1);
    }
    dilations = rewriter.getI64VectorAttr(vec);

    SmallVector<Value> poolingOps;

    ValueRange operands = adaptor.getInputs();
    ValueRange initValues = adaptor.getInitValues();
    for (auto it : llvm::zip(op.getResults(), operands, initValues)) {
      OpResult result = std::get<0>(it);
      Value input = std::get<1>(it);
      Value initValue = std::get<2>(it);
      auto resultType = cast<ShapedType>(result.getType());
      if (!cast<ShapedType>(input.getType()).getElementType().isF32()) {
        return rewriter.notifyMatchFailure(op,
                                           "expected element type to be f32");
      }

      // Create a fake window dimension.
      auto fakeWindowDims = rewriter.create<tensor::EmptyOp>(
          loc, fakeWindowShapes, resultType.getElementType());

      SmallVector<Value> resultDynamicDims;
      for (const auto &en : llvm::enumerate(resultType.getShape())) {
        if (en.value() != ShapedType::kDynamic)
          continue;
        Value dimSize = rewriter.create<tensor::DimOp>(loc, input, en.index());
        if (en.index() == 0 || static_cast<int64_t>(en.index()) == rank - 1) {
          // batch dims and channel dims can be derived from input dims
          // directly.
          resultDynamicDims.push_back(dimSize);
        } else {
          auto i = en.index() - 1;
          auto stride =
              llvm::cast<DenseIntElementsAttr>(strides).getValues<int64_t>()[i];
          auto dilation = llvm::cast<DenseIntElementsAttr>(dilations)
                              .getValues<int64_t>()[i];
          // let j = i * stride
          // output[i] = reduce( input[j, j + window_size * dilation) )
          Value offset = rewriter.create<arith::ConstantIndexOp>(
              loc, fakeWindowShapes[i] * dilation);
          dimSize = rewriter.create<arith::SubIOp>(loc, dimSize, offset);
          dimSize = rewriter.create<arith::DivUIOp>(
              loc, dimSize,
              rewriter.create<arith::ConstantIndexOp>(loc, stride));
          dimSize = rewriter.create<arith::AddIOp>(
              loc, dimSize, rewriter.create<arith::ConstantIndexOp>(loc, 1));
          resultDynamicDims.push_back(dimSize);
        }
      }
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType(),
          resultDynamicDims);

      initValue = rewriter.create<tensor::ExtractOp>(loc, initValue);
      Value filledInitTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor)
              .getResult(0);
      auto createOp = [&](auto *typePtr) -> linalg::LinalgOp {
        return cast<linalg::LinalgOp>(
            rewriter
                .create<std::remove_pointer_t<decltype(typePtr)>>(
                    loc, ArrayRef<Type>{resultType},
                    ValueRange{input, fakeWindowDims.getResult()},
                    filledInitTensor, strides, dilations,
                    linalg::getPrunedAttributeList(op))
                .getOperation());
      };
      linalg::LinalgOp poolingOp;
      PoolingType poolingType = getPoolingType(op, result.getResultNumber());
      switch (poolingType) {
      case PoolingType::k2DMin: {
        poolingOp = createOp(static_cast<linalg::PoolingNhwcMinOp *>(nullptr));
        break;
      }
      case PoolingType::k3DMin: {
        poolingOp = createOp(static_cast<linalg::PoolingNdhwcMinOp *>(nullptr));
        break;
      }
      case PoolingType::k2DMax: {
        poolingOp = createOp(static_cast<linalg::PoolingNhwcMaxOp *>(nullptr));
        break;
      }
      case PoolingType::k3DMax: {
        poolingOp = createOp(static_cast<linalg::PoolingNdhwcMaxOp *>(nullptr));
        break;
      }
      case PoolingType::k2DAdd: {
        poolingOp = createOp(static_cast<linalg::PoolingNhwcSumOp *>(nullptr));
        break;
      }
      case PoolingType::k3DAdd: {
        poolingOp = createOp(static_cast<linalg::PoolingNdhwcSumOp *>(nullptr));
        break;
      }
      case PoolingType::kInvalid:
        return rewriter.notifyMatchFailure(op, "unknown reduction operation");
      }
      poolingOps.push_back(poolingOp->getResult(0));
    }
    rewriter.replaceOp(op, poolingOps);
    return success();
  }
};

} // namespace

namespace detail {
void populateStableHloReductionToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns, bool enablePrimitiveOps) {
  if (enablePrimitiveOps) {
    patterns->add<ReduceOpToReduceConverter>(typeConverter, context);
  } else {
    patterns->add<ReduceOpToGenericConverter>(typeConverter, context);
  }
  patterns->add<ReduceRegionReturnOpConversion,
                ReduceWindowOpOnTensorsGenericConversion>(typeConverter,
                                                          context);

  // Ensure specialized patterns are higher priority than their generic
  // versions.
  patterns->add<ReduceWindowOpConversion>(typeConverter, context,
                                          PatternBenefit(2));
}
} // namespace detail
} // namespace mlir::iree_compiler::stablehlo
