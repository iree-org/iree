// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific logic for lowering StableHLO/CHLO dialects to
// LinalgExt dialect.

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>

#include "compiler/plugins/input/StableHLO/Conversion/LegalizeToLinalgUtils.h"
#include "compiler/plugins/input/StableHLO/Conversion/MapStableHLOToScalarOp.h"
#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOLINALGEXT
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

Type convertIntegerToSignless(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

std::optional<Type> convertRank0TensorToScalar(RankedTensorType tensorType) {
  if (tensorType.getRank() != 0)
    return std::nullopt;
  Type elementType = tensorType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    elementType = convertIntegerToSignless(intType);
  }
  return elementType;
}

Type convertShapedToSignless(ShapedType shapedType) {
  if (auto intType = dyn_cast<IntegerType>(shapedType.getElementType())) {
    return shapedType.clone(convertIntegerToSignless(intType));
  }
  return shapedType;
}

Value materializeCast(OpBuilder &builder, Type toType, ValueRange inputs,
                      Location loc) {
  assert(inputs.size() == 1 && "too many inputs to type conversion");
  Value fromValue = inputs[0];
  auto fromType = dyn_cast<RankedTensorType>(fromValue.getType());
  if (!fromType)
    return Value();

  if (auto intFromType = dyn_cast<IntegerType>(fromType.getElementType())) {
    Type castType = getElementTypeOrSelf(toType);
    if (auto shapedType = dyn_cast<ShapedType>(fromType)) {
      castType = shapedType.clone(castType);
    }

    if (castType != fromType) {
      fromValue =
          builder.create<UnrealizedConversionCastOp>(loc, castType, fromValue)
              ->getResult(0);
    }
  }

  if (fromType.getRank() != 0)
    return fromValue;

  Type extractType = getElementTypeOrSelf(toType);
  return builder.createOrFold<tensor::ExtractOp>(loc, extractType, fromValue);
}

/// Note: only designed to work for casts involving rank-0 tensors and scalars
/// implicitly captured within op regions.
struct StableHloToStdTypeConverter final : TypeConverter {
  StableHloToStdTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion(convertShapedToSignless);
    addConversion(convertRank0TensorToScalar);
    addConversion(convertIntegerToSignless);

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

bool isInBodyOfLinalgExtOps(Operation *op) {
  auto parent_op = op->getParentRegion()->getParentOp();
  return parent_op->getDialect() ==
         parent_op->getContext()
             ->getLoadedDialect<IREE::LinalgExt::IREELinalgExtDialect>();
}

//===----------------------------------------------------------------------===//
// Region operations lowering.
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct LinalgExtRegionHLOOpConversion final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isInBodyOfLinalgExtOps(op))
      return failure();
    TensorType origRetType = dyn_cast<TensorType>(op.getType());
    if (!origRetType)
      return failure();
    SmallVector<Value> scalarArgs;
    Type newRetType = getElementTypeOrSelf(
        this->typeConverter->convertType(origRetType.getElementType()));
    Value result = mlir::stablehlo::StableHloOpToStdScalarOp::mapOp(
        op, newRetType, adaptor.getOperands(), &rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinalgExtRegionReturnOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isInBodyOfLinalgExtOps(op))
      return failure();
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::YieldOp>(
        op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

struct SortOpConversion final : OpConversionPattern<mlir::stablehlo::SortOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::SortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    llvm::SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    };
    auto sortOp = rewriter.create<IREE::LinalgExt::SortOp>(
        loc, resultTypes,
        /*inputs=*/ValueRange{}, adaptor.getOperands(), op.getDimensionAttr());
    rewriter.inlineRegionBefore(op.getComparator(), sortOp.getRegion(),
                                sortOp.getRegion().begin());
    Region &region = sortOp.getRegion();
    Block &block = region.front();
    TypeConverter::SignatureConversion signature_converter(
        block.getNumArguments());
    for (auto [idx, argument] : llvm::enumerate(block.getArguments())) {
      signature_converter.addInputs(
          idx, getTypeConverter()->convertType(
                   getElementTypeOrSelf(argument.getType())));
    }
    rewriter.applySignatureConversion(&region.front(), signature_converter);

    rewriter.replaceOp(op, sortOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

struct ScatterOpConversion final
    : OpConversionPattern<mlir::stablehlo::ScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  /// Returns true if the `dimensionNumbers` from the stablehlo.scatter op
  /// follows a canonical form:
  ///
  /// * The rank of indices is greater than or equal to two.
  /// * The index_vector_dim is the last dim of indices.
  /// * Scatter dims to operand dims order: (0, ... , n)
  /// * Inserted window dims order: (0, ... , d)
  /// * Update window dims order: (d + 1, ... , m)
  static bool hasCanonicalDimensionNumbers(mlir::stablehlo::ScatterOp op) {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indicesType = llvm::cast<ShapedType>(op.getScatterIndices().getType());
    auto indicesRank = indicesType.getRank();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indexDepth = indicesType.getShape().back();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();

    if (indicesRank != 2)
      return false;
    if (indexVectorDim != indicesRank - 1)
      return false;
    if (scatterDimsToOperandDims.size() != indexDepth)
      return false;

    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    for (auto [idx, dim] : llvm::enumerate(insertedWindowDims)) {
      if (idx != dim)
        return false;
    }

    // Check that there is only one batch dimension in the updates.
    for (auto [idx, dim] : llvm::enumerate(dimNumbers.getUpdateWindowDims())) {
      if (idx + 1 != dim)
        return false;
    }

    return true;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasCanonicalDimensionNumbers(op))
      return failure();
    if (llvm::size(op.getInputs()) != 1)
      return op.emitError("NYI variadic operands scatter");
    if (llvm::size(op.getUpdates()) != 1)
      return op.emitError("NYI variadic updates scatter");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value original = adaptor.getInputs().front();
    Value indices = adaptor.getScatterIndices();
    Value updates = adaptor.getUpdates().front();

    auto originalType = llvm::dyn_cast<ShapedType>(original.getType());

    llvm::SmallVector<int64_t> scatterDimMap;
    for (auto dim :
         op.getScatterDimensionNumbers().getScatterDimsToOperandDims()) {
      scatterDimMap.push_back(dim);
    }

    auto scatterOp = rewriter.create<IREE::LinalgExt::ScatterOp>(
        op.getLoc(), originalType, ValueRange{updates, indices},
        ValueRange{original}, scatterDimMap, op.getUniqueIndices());

    rewriter.inlineRegionBefore(op.getUpdateComputation(),
                                scatterOp.getRegion(),
                                scatterOp.getRegion().begin());
    Region &region = scatterOp.getRegion();
    TypeConverter::SignatureConversion signatureConverter(2);
    Type argType = getElementTypeOrSelf(original.getType());
    // stablehlo.scatter op takes:
    //   output[O] = update_computation(output[O], updates[U])
    // where output[O] maps to block args #1 in linalg_ext.scatter ops.
    signatureConverter.addInputs(1, argType);
    signatureConverter.addInputs(0, argType);
    rewriter.applySignatureConversion(&region.front(), signatureConverter);

    rewriter.replaceOp(op, scatterOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

struct FftOpConversion final : OpConversionPattern<mlir::stablehlo::FftOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::FftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle 2^n fft length.
    if (!llvm::all_equal(op.getFftLength())) {
      return rewriter.notifyMatchFailure(op, "non-splat length");
    }
    int64_t fftLength = op.getFftLength().front();
    if (!llvm::isPowerOf2_64(fftLength)) {
      return rewriter.notifyMatchFailure(
          op, "expected FFT length to be a power of two");
    }

    auto rewriteRes = IREE::LinalgExt::rewriteFft(op, adaptor.getOperand(),
                                                  fftLength, rewriter);
    if (failed(rewriteRes)) {
      return failure();
    }

    auto [real, imag] = rewriteRes.value();

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ComplexOp>(op, op.getType(),
                                                            real, imag);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

struct ReverseOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReverseOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    if (!ty)
      return failure();

    Value input = op.getOperand();
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<ShapedType>(op.getType());
    ArrayRef<int64_t> dims = op.getDimensions();
    Location loc = op.getLoc();
    int64_t inputTyRank = inputTy.getRank();

    // First fill the output buffer with the init value.
    SmallVector<OpFoldResult> inputMixedSizes =
        tensor::getMixedSizes(rewriter, loc, input);
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, inputMixedSizes, inputTy.getElementType());
    SmallVector<AffineMap> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, ArrayRef<Value>({}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          llvm::SmallVector<Value> indices;
          for (unsigned int i = 0; i < inputTyRank; i++) {
            Value index =
                rewriter.create<linalg::IndexOp>(nestedLoc, i).getResult();
            if (std::find(dims.begin(), dims.end(), i) != dims.end()) {
              auto one = rewriter.create<arith::ConstantIndexOp>(nestedLoc, 1);
              Value axisDimSize = rewriter.create<tensor::DimOp>(loc, input, i);
              auto sizeMinusOne =
                  rewriter.create<arith::SubIOp>(nestedLoc, axisDimSize, one);
              index = rewriter.create<arith::SubIOp>(nestedLoc, sizeMinusOne,
                                                     index);
            }
            indices.push_back(index);
          }

          auto extract = nestedBuilder.create<tensor::ExtractOp>(
              nestedLoc, input, indices);
          nestedBuilder.create<linalg::YieldOp>(op.getLoc(),
                                                extract.getResult());
        });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

static bool checkUnary(const ArrayRef<int64_t> &values) {
  for (auto value : values) {
    if (value != 1) {
      return false;
    }
  }
  return true;
}

struct ScanOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getWindowStrides() && !checkUnary(*op.getWindowStrides())) {
      return rewriter.notifyMatchFailure(op, "non-unary stride");
    }

    if (op.getWindowDilations() && !checkUnary(*op.getWindowDilations())) {
      return rewriter.notifyMatchFailure(op, "non-unary window dilations");
    }

    if (op.getBaseDilations() && !checkUnary(*op.getBaseDilations())) {
      return rewriter.notifyMatchFailure(op, "non-unary base dilations");
    }

    auto inputs = op.getInputs();
    if (inputs.size() != 1) {
      return rewriter.notifyMatchFailure(op, "more than one input");
    }

    auto input0 = inputs.front();
    auto input0Ty = cast<ShapedType>(input0.getType());
    auto init0 = op.getInitValues().front();
    auto init0Ty = cast<ShapedType>(init0.getType());

    auto window = llvm::to_vector(op.getWindowDimensions());
    llvm::SmallVector<int64_t, 4> reduceAxes;
    for (int i = 0, s = window.size(); i < s; ++i) {
      if (window[i] == 1)
        continue;
      if (window[i] == input0Ty.getDimSize(i)) {
        reduceAxes.push_back(i);
        continue;
      }

      // Arguably it's still beneficial across a partial window, but this
      // depends on performance characteristics.
      return rewriter.notifyMatchFailure(op, "not length-1 or full width");
    }

    if (reduceAxes.size() != 1) {
      return rewriter.notifyMatchFailure(op, "non singular reduction axis");
    }

    const int64_t reduceAxis = reduceAxes.front();

    if (!op.getPadding()) {
      return rewriter.notifyMatchFailure(op, "no padding values found");
    }

    auto padding = extract1DVector(*op.getPadding());
    if (padding.size() < reduceAxis * 2) {
      return rewriter.notifyMatchFailure(op, "no padding along reduction");
    }

    for (int i = 0, s = padding.size(); i < s; i += 2) {
      if (i == reduceAxis * 2)
        continue;
      if (padding[i] != 0 || padding[i + 1] != 0) {
        return rewriter.notifyMatchFailure(op,
                                           "padding along non-reduction axis");
      }
    }

    bool isPrefix =
        padding[reduceAxis * 2] == (input0Ty.getDimSize(reduceAxis) - 1);
    bool isPostfix =
        padding[reduceAxis * 2 + 1] == (input0Ty.getDimSize(reduceAxis) - 1);

    if (isPrefix == isPostfix) {
      return rewriter.notifyMatchFailure(op, "is not purely prefix or postfix");
    }

    llvm::SmallVector<Value> outputs;
    llvm::SmallVector<Value> outputDynDims;
    for (int i = 0; i < input0Ty.getRank(); ++i) {
      if (input0Ty.isDynamic(i)) {
        outputDynDims.push_back(
            rewriter.createOrFold<tensor::DimOp>(op.getLoc(), input0, i));
      }
    }

    llvm::SmallVector<Value> init;
    llvm::SmallVector<int64_t> initDims;
    llvm::SmallVector<Value> initDynDims;
    for (int i = 0; i < input0Ty.getRank(); ++i) {
      if (i == reduceAxis)
        continue;
      initDims.push_back(input0Ty.getDimSize(i));
      if (ShapedType::isDynamic(initDims.back())) {
        initDynDims.push_back(
            rewriter.createOrFold<tensor::DimOp>(op.getLoc(), input0, i));
      }
    }

    outputs.push_back(rewriter.create<tensor::EmptyOp>(
        op.getLoc(), input0Ty.getShape(), input0Ty.getElementType(),
        outputDynDims));

    Value newInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), initDims, init0Ty.getElementType(), initDynDims);

    SmallVector<AffineMap> indexingMaps{
        AffineMap::get(initDims.size(), /*symbolCount=*/0, {},
                       rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(initDims.size())};
    SmallVector<utils::IteratorType> iterators(initDims.size(),
                                               utils::IteratorType::parallel);

    newInit = rewriter
                  .create<linalg::GenericOp>(
                      op.getLoc(), init0Ty.clone(initDims), ValueRange{init0},
                      ValueRange{newInit}, indexingMaps, iterators,
                      [&](OpBuilder &b, Location loc, ValueRange args) {
                        b.create<linalg::YieldOp>(loc, args[0]);
                      })
                  .getResult(0);
    outputs.push_back(newInit);

    llvm::SmallVector<Type> outputTys;
    for (auto output : outputs) {
      outputTys.push_back(output.getType());
    }

    auto scanOp = rewriter.create<IREE::LinalgExt::ScanOp>(
        op.getLoc(), outputTys, inputs, outputs,
        rewriter.getI64IntegerAttr(reduceAxis), rewriter.getBoolAttr(1));

    rewriter.inlineRegionBefore(op.getRegion(), scanOp.getRegion(),
                                scanOp.getRegion().begin());

    // Handle the tensor<*> to * conversion:
    TypeConverter::SignatureConversion signatureConverter(2);
    signatureConverter.addInputs(0, input0Ty.getElementType());
    signatureConverter.addInputs(1, init0Ty.getElementType());
    rewriter.applySignatureConversion(&scanOp.getRegion().front(),
                                      signatureConverter);

    rewriter.replaceOp(op, scanOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

struct TopkOpConversion final : OpConversionPattern<chlo::TopKOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(chlo::TopKOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();

    auto inputValuesType = llvm::dyn_cast<ShapedType>(operand.getType());
    auto outputValuesType =
        llvm::dyn_cast<ShapedType>(op.getValues().getType());
    auto outputIndicesType =
        llvm::dyn_cast<ShapedType>(op.getIndices().getType());
    if (!inputValuesType || !outputValuesType || !outputIndicesType) {
      return rewriter.notifyMatchFailure(
          op, "Input and output must be of ShapedType");
    }

    Type valueElementType = inputValuesType.getElementType();
    Type indicesElementType = outputIndicesType.getElementType();
    // Only handle integer types for indicies. Index type is not supported.
    if (!llvm::isa<IntegerType>(indicesElementType)) {
      return rewriter.notifyMatchFailure(
          op, "Output indices must be of integer type.");
    }

    // Create and initialize output tensors for LinalgExt TopK results
    // Define the output types based on the results of CHLO TopK
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getOperand());
    mixedSizes.back() = rewriter.getIndexAttr(adaptor.getK());
    Value emptyTensorOutputValues = rewriter.create<mlir::tensor::EmptyOp>(
        loc, mixedSizes, valueElementType);
    Value emptyTensorOutputIndices = rewriter.create<mlir::tensor::EmptyOp>(
        loc, mixedSizes, indicesElementType);
    // Initialize indices to 0 and values to negative infinity
    TypedAttr negInfAttr;
    if (auto intType = llvm::dyn_cast<IntegerType>(valueElementType)) {
      negInfAttr = rewriter.getIntegerAttr(
          intType, APInt::getSignedMinValue(intType.getWidth()));
    } else {
      auto negApFloat = APFloat::getInf(
          llvm::cast<FloatType>(valueElementType).getFloatSemantics(),
          /*Negative=*/true);
      negInfAttr = rewriter.getFloatAttr(valueElementType, negApFloat);
    }
    Value negInf = rewriter.create<arith::ConstantOp>(loc, negInfAttr);
    TypedAttr posInfAttr = rewriter.getIntegerAttr(
        indicesElementType, APInt::getSignedMaxValue(32));
    Value posInf = rewriter.create<arith::ConstantOp>(loc, posInfAttr);
    Value negInfTensor =
        rewriter.create<linalg::FillOp>(loc, negInf, emptyTensorOutputValues)
            .result();
    Value posInfTensor =
        rewriter.create<linalg::FillOp>(loc, posInf, emptyTensorOutputIndices)
            .result();

    // Replace the CHLO TopK with LinalgExt TopK
    uint64_t kDim = inputValuesType.getRank() - 1;
    SmallVector<Type> newResultTypes;
    newResultTypes.push_back(outputValuesType.cloneWith(
        outputValuesType.getShape(), valueElementType));
    for (int i = 1; i < op->getResultTypes().size(); i++) {
      newResultTypes.push_back(op->getResultTypes()[i]);
    }
    auto topkOp = rewriter.replaceOpWithNewOp<IREE::LinalgExt::TopkOp>(
        op, newResultTypes, ValueRange{operand},
        ValueRange{negInfTensor, posInfTensor}, kDim);

    // Define the region of TopK with a GT comparison
    SmallVector<Type> types(2, valueElementType);
    SmallVector<Location> locations(2, loc);
    Block *block =
        rewriter.createBlock(&topkOp.getRegion(), {}, types, locations);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      Value lhs = block->getArgument(0);
      Value rhs = block->getArgument(1);
      Value condition;
      if (llvm::isa<IntegerType>(valueElementType)) {
        condition = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, lhs, rhs);
      } else {
        condition = rewriter.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::OGT, lhs, rhs);
      }
      rewriter.create<IREE::LinalgExt::YieldOp>(loc, condition);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct ConvertStableHloToLinalgExt final
    : impl::ConvertStableHloToLinalgExtBase<ConvertStableHloToLinalgExt> {
  using ConvertStableHloToLinalgExtBase::ConvertStableHloToLinalgExtBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect, linalg::LinalgDialect,
                IREE::Flow::FlowDialect, mlir::cf::ControlFlowDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect,
                complex::ComplexDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    StableHloToStdTypeConverter typeConverter;
    populateStableHloToLinalgExtConversionPatterns(context, typeConverter,
                                                   &patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect,
                           linalg::LinalgDialect, IREE::Flow::FlowDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::math::MathDialect, mlir::arith::ArithDialect,
                           tensor::TensorDialect, complex::ComplexDialect>();
    // TODO: Scatter is not marked as illegal to allow falling back to the
    // generic LinAlg lowering, the generic lowering is not always performant
    // and even though only used in fallback here, may hide performance
    // issues and we'd rather know when the optimized lowering fails.
    target.addIllegalOp<mlir::stablehlo::SortOp, mlir::stablehlo::FftOp,
                        mlir::stablehlo::ReverseOp>();
    // FFT conversion creates complex ops which will be converted by the normal
    // StableHlo lowering, but these should still be converted if present inside
    // other linalg_ext op regions.
    target.addDynamicallyLegalOp<mlir::stablehlo::ComplexOp>(
        [](mlir::stablehlo::ComplexOp complexOp) {
          return !isInBodyOfLinalgExtOps(complexOp);
        });
    // We deliberately allow unrealized casts to persist. These should fall away
    // when the rest of StableHlo is converted.
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

void populateStableHloToLinalgExtConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns) {
  patterns->add<ScanOpConversion, SortOpConversion, ScatterOpConversion,
                FftOpConversion, ReverseOpConversion, TopkOpConversion>(
      typeConverter, context);

  // FIXME: It shouldn't be necessary to list every matching StableHlo op
  // here, especially since they're already listed in
  // populateStableHloToLinalgConversionPattern and in
  // StableHloOpToStdScalarOp. These lists are all the same. Can we leverage
  // SFINAE here?
  patterns->add<
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::AbsOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::AddOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::AndOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::Atan2Op>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::BitcastConvertOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::CeilOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ClampOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::CompareOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ComplexOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ConvertOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::CosineOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::DivOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ExpOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::Expm1Op>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::FloorOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ImagOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::IsFiniteOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::LogOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::LogisticOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::Log1pOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::MaxOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::MinOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::MulOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::NegOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::NotOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::OrOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::PowOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::RealOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::RemOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::RsqrtOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::SelectOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ShiftLeftOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ShiftRightArithmeticOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::ShiftRightLogicalOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::SignOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::SineOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::SqrtOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::SubtractOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::TanhOp>,
      LinalgExtRegionHLOOpConversion<mlir::stablehlo::XorOp>,
      LinalgExtRegionReturnOpConversion>(typeConverter, context);
}

} // namespace mlir::iree_compiler::stablehlo
