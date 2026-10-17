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
#include "llvm/ADT/Repeated.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOLINALGEXT
#define GEN_PASS_DEF_LEGALIZEDYNAMICREDUCEWINDOW
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

Type convertIntegerToSignless(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

std::optional<Type> convertRank0TensorToScalar(RankedTensorType tensorType) {
  if (tensorType.getRank() != 0) {
    return std::nullopt;
  }
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
  if (!fromType) {
    auto toTensorType = dyn_cast<RankedTensorType>(toType);
    if (!toTensorType || toTensorType.getRank() != 0) {
      return Value();
    }
    fromType = toTensorType.clone(fromValue.getType());
    fromValue = tensor::FromElementsOp::create(builder, loc, fromType, inputs);
  }

  if (auto intFromType = dyn_cast<IntegerType>(fromType.getElementType())) {
    Type castType = getElementTypeOrSelf(toType);
    if (auto shapedType = dyn_cast<ShapedType>(fromType)) {
      castType = shapedType.clone(castType);
    }

    if (castType != fromType) {
      fromValue =
          UnrealizedConversionCastOp::create(builder, loc, castType, fromValue)
              ->getResult(0);
    }
  }

  if (fromType.getRank() != 0 || isa<ShapedType>(toType)) {
    return fromValue;
  }

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
    if (!isInBodyOfLinalgExtOps(op)) {
      return failure();
    }
    TensorType origRetType = dyn_cast<TensorType>(op.getType());
    if (!origRetType) {
      return failure();
    }
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
  using Base::Base;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isInBodyOfLinalgExtOps(op)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::YieldOp>(
        op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

struct SortOpConversion final : OpConversionPattern<mlir::stablehlo::SortOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::SortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    llvm::SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    };
    auto sortOp = IREE::LinalgExt::SortOp::create(
        rewriter, loc, resultTypes,
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
  using Base::Base;

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
    auto indicesType = cast<ShapedType>(op.getScatterIndices().getType());
    auto indicesRank = indicesType.getRank();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indexDepth = indicesType.getShape().back();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();

    if (indicesRank != 2) {
      return false;
    }
    if (indexVectorDim != indicesRank - 1) {
      return false;
    }
    if (scatterDimsToOperandDims.size() != indexDepth) {
      return false;
    }
    for (auto [idx, dim] : llvm::enumerate(scatterDimsToOperandDims)) {
      if (idx != dim) {
        return false;
      }
    }

    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    for (auto [idx, dim] : llvm::enumerate(insertedWindowDims)) {
      if (idx != dim) {
        return false;
      }
    }

    // Check that there is only one batch dimension in the updates.
    for (auto [idx, dim] : llvm::enumerate(dimNumbers.getUpdateWindowDims())) {
      if (idx + 1 != dim) {
        return false;
      }
    }

    return true;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasCanonicalDimensionNumbers(op)) {
      return failure();
    }
    if (llvm::size(op.getInputs()) != 1) {
      return op.emitError("NYI variadic operands scatter");
    }
    if (llvm::size(op.getUpdates()) != 1) {
      return op.emitError("NYI variadic updates scatter");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value original = adaptor.getInputs().front();
    Value indices = adaptor.getScatterIndices();
    Value updates = adaptor.getUpdates().front();

    auto originalType = dyn_cast<ShapedType>(original.getType());

    llvm::SmallVector<int64_t> scatterDimMap;
    for (auto dim :
         op.getScatterDimensionNumbers().getScatterDimsToOperandDims()) {
      scatterDimMap.push_back(dim);
    }

    auto scatterOp = IREE::LinalgExt::ScatterOp::create(
        rewriter, op.getLoc(), originalType, updates, indices,
        /*mask=*/Value(), original, scatterDimMap, op.getUniqueIndices());

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
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::FftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Only handle 1D ffts.
    if (op.getFftLength().size() != 1) {
      return rewriter.notifyMatchFailure(op, "only 1D FFTs supported");
    }
    int64_t fftLength = op.getFftLength().front();
    if (!llvm::isPowerOf2_64(fftLength)) {
      return rewriter.notifyMatchFailure(
          op, "expected FFT length to be a power of two");
    }
    mlir::stablehlo::FftType fftType = op.getFftType();
    FailureOr<std::pair<Value, Value>> rewriteRes;
    if (fftType == mlir::stablehlo::FftType::RFFT) {
      rewriteRes = IREE::LinalgExt::rewriteRfft(op, adaptor.getOperand(),
                                                fftLength, rewriter);
    } else if (fftType == mlir::stablehlo::FftType::FFT) {
      rewriteRes = IREE::LinalgExt::rewriteFft(
          op, adaptor.getOperand(), fftLength,
          IREE::LinalgExt::FFTDirection::Forward,
          IREE::LinalgExt::FFTNormalization::NoNormalize, rewriter);
    } else if (fftType == mlir::stablehlo::FftType::IFFT) {
      rewriteRes = IREE::LinalgExt::rewriteFft(
          op, adaptor.getOperand(), fftLength,
          IREE::LinalgExt::FFTDirection::Backward,
          IREE::LinalgExt::FFTNormalization::Normalize, rewriter);
    } else if (fftType == mlir::stablehlo::FftType::IRFFT) {
      rewriteRes = IREE::LinalgExt::rewriteIrfft(op, adaptor.getOperand(),
                                                 fftLength, rewriter);
    } else {
      // Not implemented.
      return failure();
    }

    if (failed(rewriteRes)) {
      return failure();
    }

    auto [real, imag] = rewriteRes.value();

    // For IRFFT, return only the real part (output is real-valued)
    if (fftType == mlir::stablehlo::FftType::IRFFT) {
      rewriter.replaceOp(op, real);
    } else {
      // For other FFT types, return complex output
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ComplexOp>(op, op.getType(),
                                                              real, imag);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

struct ReverseOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReverseOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    if (!ty) {
      return failure();
    }

    Value input = op.getOperand();
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<ShapedType>(op.getType());
    ArrayRef<int64_t> dims = op.getDimensions();
    Location loc = op.getLoc();
    int64_t inputTyRank = inputTy.getRank();

    // First fill the output buffer with the init value.
    SmallVector<OpFoldResult> inputMixedSizes =
        tensor::getMixedSizes(rewriter, loc, input);
    auto emptyTensor = tensor::EmptyOp::create(rewriter, loc, inputMixedSizes,
                                               inputTy.getElementType());
    SmallVector<AffineMap> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, ArrayRef<Value>({}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          llvm::SmallVector<Value> indices;
          for (unsigned int i = 0; i < inputTyRank; i++) {
            Value index =
                linalg::IndexOp::create(rewriter, nestedLoc, i).getResult();
            if (std::find(dims.begin(), dims.end(), i) != dims.end()) {
              auto one = arith::ConstantIndexOp::create(rewriter, nestedLoc, 1);
              Value axisDimSize =
                  tensor::DimOp::create(rewriter, loc, input, i);
              auto sizeMinusOne =
                  arith::SubIOp::create(rewriter, nestedLoc, axisDimSize, one);
              index = arith::SubIOp::create(rewriter, nestedLoc, sizeMinusOne,
                                            index);
            }
            indices.push_back(index);
          }

          auto extract = tensor::ExtractOp::create(nestedBuilder, nestedLoc,
                                                   input, indices);
          linalg::YieldOp::create(nestedBuilder, op.getLoc(),
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

// Returns true when `init` is the identity of the reducer `body`, one
// elementwise operation on its two arguments; an inclusive scan never combines
// the init.
static bool isReducerIdentity(Block &body, Value init) {
  Operation *reduceOp = body.getTerminator()->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getBlock() != &body || &body.front() != reduceOp ||
      reduceOp->getNumOperands() != 2 ||
      !llvm::is_contained(body.getArguments(), reduceOp->getOperand(0)) ||
      !llvm::is_contained(body.getArguments(), reduceOp->getOperand(1)) ||
      reduceOp->getOperand(0) == reduceOp->getOperand(1)) {
    return false;
  }
  DenseElementsAttr value;
  if (!matchPattern(init, m_Constant(&value)) || !value.isSplat()) {
    return false;
  }
  if (isa<FloatType>(value.getElementType())) {
    APFloat x = value.getSplatValue<APFloat>();
    return TypeSwitch<Operation *, bool>(reduceOp)
        .Case([&](mlir::stablehlo::AddOp) { return x.isZero(); })
        .Case([&](mlir::stablehlo::MulOp) { return x.isExactlyValue(1.0); })
        .Case([&](mlir::stablehlo::MaxOp) {
          return x.isInfinity() && x.isNegative();
        })
        .Case([&](mlir::stablehlo::MinOp) {
          return x.isInfinity() && !x.isNegative();
        })
        .Default(false);
  }
  if (!value.getElementType().isIntOrIndex()) {
    return false;
  }
  APInt x = value.getSplatValue<APInt>();
  bool isUnsigned = value.getElementType().isUnsignedInteger();
  return TypeSwitch<Operation *, bool>(reduceOp)
      .Case<mlir::stablehlo::AddOp, mlir::stablehlo::OrOp,
            mlir::stablehlo::XorOp>([&](auto) { return x.isZero(); })
      .Case([&](mlir::stablehlo::MulOp) { return x.isOne(); })
      .Case([&](mlir::stablehlo::AndOp) { return x.isAllOnes(); })
      .Case([&](mlir::stablehlo::MaxOp) {
        return isUnsigned ? x.isZero() : x.isMinSignedValue();
      })
      .Case([&](mlir::stablehlo::MinOp) {
        return isUnsigned ? x.isAllOnes() : x.isMaxSignedValue();
      })
      .Default(false);
}

// Builds an inclusive scan of `input` along `reduceAxis` seeded with `init`;
// a suffix scan reverses around it. `fillReducer` moves or clones the reducer
// into the scan region: a block of two 0-d tensors ending in stablehlo.return.
static Value
createReduceWindowScan(ConversionPatternRewriter &rewriter, Location loc,
                       Value input, Value init, int64_t reduceAxis,
                       bool isPostfix,
                       llvm::function_ref<void(Region &)> fillReducer) {
  auto inputTy = cast<ShapedType>(input.getType());
  auto initTy = cast<ShapedType>(init.getType());
  llvm::SmallVector<Value> outputs;
  llvm::SmallVector<Value> outputDynDims;
  for (int i = 0; i < inputTy.getRank(); ++i) {
    if (inputTy.isDynamicDim(i)) {
      outputDynDims.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, input, i));
    }
  }

  llvm::SmallVector<int64_t> initDims;
  llvm::SmallVector<Value> initDynDims;
  for (int i = 0; i < inputTy.getRank(); ++i) {
    if (i == reduceAxis) {
      continue;
    }
    initDims.push_back(inputTy.getDimSize(i));
    if (ShapedType::isDynamic(initDims.back())) {
      initDynDims.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, input, i));
    }
  }

  outputs.push_back(tensor::EmptyOp::create(rewriter, loc, inputTy.getShape(),
                                            inputTy.getElementType(),
                                            outputDynDims));

  Value newInit = tensor::EmptyOp::create(rewriter, loc, initDims,
                                          initTy.getElementType(), initDynDims);

  SmallVector<AffineMap> indexingMaps{
      AffineMap::get(initDims.size(), /*symbolCount=*/0, {},
                     rewriter.getContext()),
      rewriter.getMultiDimIdentityMap(initDims.size())};
  SmallVector<utils::IteratorType> iterators(initDims.size(),
                                             utils::IteratorType::parallel);

  newInit = linalg::GenericOp::create(
                rewriter, loc, initTy.clone(initDims), ValueRange{init},
                ValueRange{newInit}, indexingMaps, iterators,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  linalg::YieldOp::create(b, loc, args[0]);
                })
                .getResult(0);
  outputs.push_back(newInit);

  llvm::SmallVector<Type> outputTys;
  for (auto output : outputs) {
    outputTys.push_back(output.getType());
  }

  // IREE::LinalgExt::ScanOp only supports purely inclusive (prefix) scans.
  // To support suffix (postfix) scans, we can reverse the input, perform a
  // prefix scan, and then reverse the output.
  auto createReverse = [&](Value val, int64_t axis) -> Value {
    auto valTy = llvm::cast<mlir::RankedTensorType>(val.getType());
    SmallVector<int64_t> dimensions;
    dimensions.push_back(axis);
    return mlir::stablehlo::ReverseOp::create(
        rewriter, loc, valTy, val, rewriter.getDenseI64ArrayAttr(dimensions));
  };
  Value scanInput = isPostfix ? createReverse(input, reduceAxis) : input;

  auto scanOp = IREE::LinalgExt::ScanOp::create(
      rewriter, loc, outputTys, scanInput, outputs[0], outputs[1],
      rewriter.getI64IntegerAttr(reduceAxis), rewriter.getBoolAttr(true));

  fillReducer(scanOp.getRegion());

  // Handle the tensor<*> to * conversion:
  TypeConverter::SignatureConversion signatureConverter(2);
  signatureConverter.addInputs(0, inputTy.getElementType());
  signatureConverter.addInputs(1, initTy.getElementType());
  rewriter.applySignatureConversion(&scanOp.getRegion().front(),
                                    signatureConverter);

  Value result = scanOp.getResult(0);
  if (isPostfix) {
    result = createReverse(result, reduceAxis);
  }
  return result;
}

struct ScanOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using Base::Base;
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
    if (!isReducerIdentity(op.getBody().front(), init0)) {
      return rewriter.notifyMatchFailure(op, "init is not the identity");
    }

    auto window = llvm::to_vector(op.getWindowDimensions());
    llvm::SmallVector<int64_t, 4> reduceAxes;
    for (int i = 0, s = window.size(); i < s; ++i) {
      if (window[i] == 1) {
        continue;
      }
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
      if (i == reduceAxis * 2) {
        continue;
      }
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

    Value result = createReduceWindowScan(
        rewriter, op.getLoc(), input0, init0, reduceAxis, isPostfix,
        [&](Region &region) {
          rewriter.inlineRegionBefore(op.getRegion(), region, region.begin());
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// stablehlo.dynamic_reduce_window
//===----------------------------------------------------------------------===//

// An element of a shape operand: a constant, or the scalar holding it.
struct ShapeElement {
  std::optional<int64_t> constant;
  Value value;
};

// Returns element `index` of the integer tensor `tensor` when only constants
// and data movement build it.
static std::optional<ShapeElement> getShapeElement(Value tensor,
                                                   SmallVector<int64_t> index) {
  auto type = dyn_cast<RankedTensorType>(tensor.getType());
  if (!type || !type.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t linear = linearize(index, computeStrides(type.getShape()));
  DenseIntElementsAttr constant;
  if (matchPattern(tensor, m_Constant(&constant))) {
    return ShapeElement{constant.getValues<APInt>()[linear].getSExtValue(), {}};
  }
  if (auto fromElements = tensor.getDefiningOp<tensor::FromElementsOp>()) {
    Value scalar = fromElements.getElements()[linear];
    APInt value;
    if (matchPattern(scalar, m_ConstantInt(&value))) {
      return ShapeElement{value.getSExtValue(), {}};
    }
    return ShapeElement{std::nullopt, scalar};
  }
  if (auto concat = tensor.getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
    int64_t dim = concat.getDimension();
    for (Value input : concat.getInputs()) {
      int64_t size = cast<ShapedType>(input.getType()).getDimSize(dim);
      if (ShapedType::isDynamic(size)) {
        return std::nullopt;
      }
      if (index[dim] < size) {
        return getShapeElement(input, index);
      }
      index[dim] -= size;
    }
    return std::nullopt;
  }
  if (auto reshape = tensor.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto operandType = cast<RankedTensorType>(reshape.getOperand().getType());
    if (!operandType.hasStaticShape()) {
      return std::nullopt;
    }
    return getShapeElement(
        reshape.getOperand(),
        delinearize(linear, computeStrides(operandType.getShape())));
  }
  return std::nullopt;
}

// Strips casts and assumptions under which an i32 dimension size is exact.
static Value stripExactSizeCasts(Value value) {
  auto isWide = [](Type type) {
    return type.isIndex() || type.getIntOrFloatBitWidth() >= 32;
  };
  while (Operation *op = value.getDefiningOp()) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp>(op) ||
        (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::TruncIOp>(op) &&
         isWide(op->getResult(0).getType()))) {
      value = op->getOperand(0);
    } else if (auto assume = dyn_cast<IREE::Util::AssumeIntOp>(op)) {
      value = assume.getOperand(cast<OpResult>(value).getResultNumber());
    } else {
      break;
    }
  }
  return value;
}

// Returns true when `element` is the extent of dimension `dim` of `tensor`.
static bool isDimSize(const ShapeElement &element, Value tensor, int64_t dim) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (element.constant) {
    return type.getDimSize(dim) == *element.constant;
  }
  auto stripTieShape = [](Value value) {
    while (auto tie = value.getDefiningOp<IREE::Flow::TensorTieShapeOp>()) {
      value = tie.getOperand();
    }
    return value;
  };
  Value size = stripExactSizeCasts(element.value);
  if (auto dimOp = size.getDefiningOp<tensor::DimOp>()) {
    return dimOp.getConstantIndex() == dim &&
           stripTieShape(dimOp.getSource()) == stripTieShape(tensor);
  }
  auto tie = tensor.getDefiningOp<IREE::Flow::TensorTieShapeOp>();
  return tie && type.isDynamicDim(dim) &&
         stripExactSizeCasts(
             tie.getDynamicDims()[type.getDynamicDimIndex(dim)]) == size;
}

// Returns true when `element` is one less than the extent of dimension `dim`.
static bool isDimSizeMinusOne(const ShapeElement &element, Value tensor,
                              int64_t dim) {
  if (element.constant) {
    int64_t size = cast<RankedTensorType>(tensor.getType()).getDimSize(dim);
    return ShapedType::isStatic(size) && size - 1 == *element.constant;
  }
  auto isConstant = [](Value value, int64_t expected) {
    APInt constant;
    return matchPattern(value, m_ConstantInt(&constant)) &&
           constant.getSExtValue() == expected;
  };
  Value value = stripExactSizeCasts(element.value);
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    for (auto [size, offset] : {std::pair(add.getLhs(), add.getRhs()),
                                std::pair(add.getRhs(), add.getLhs())}) {
      if (isConstant(offset, -1) &&
          isDimSize({std::nullopt, size}, tensor, dim)) {
        return true;
      }
    }
  }
  auto sub = value.getDefiningOp<arith::SubIOp>();
  return sub && isConstant(sub.getRhs(), 1) &&
         isDimSize({std::nullopt, sub.getLhs()}, tensor, dim);
}

// The scan JAX emits as stablehlo.dynamic_reduce_window for a cumulative
// reduction over a symbolic axis.
struct DynamicReduceWindowScan {
  int64_t axis;
  bool isPostfix;
  func::FuncOp reducer;
};

// Matches one input with unit strides and dilations, a window spanning one
// axis and of size one elsewhere, and `size - 1` padding on one side of that
// axis only.
static std::optional<DynamicReduceWindowScan>
matchDynamicReduceWindowScan(mlir::stablehlo::CustomCallOp op) {
  if (op.getCallTargetName() != "stablehlo.dynamic_reduce_window" ||
      op.getNumOperands() != 7 || op.getNumResults() != 1 ||
      op.getCalledComputations().size() != 1) {
    return std::nullopt;
  }
  Value input = op.getOperand(0);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto initType = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  if (!inputType || !initType || initType.getRank() != 0 ||
      initType.getElementType() != inputType.getElementType() ||
      op.getResult(0).getType() != inputType) {
    return std::nullopt;
  }
  auto reducer = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      op, cast<FlatSymbolRefAttr>(op.getCalledComputations()[0]));
  if (!reducer || reducer.isExternal() || !reducer.getBody().hasOneBlock() ||
      reducer.getFunctionType() != FunctionType::get(op.getContext(),
                                                     {initType, initType},
                                                     {initType})) {
    return std::nullopt;
  }

  int64_t rank = inputType.getRank();
  auto hasShape = [](Value value, ArrayRef<int64_t> shape) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getShape() == shape &&
           type.getElementType().isSignlessInteger();
  };
  if (!llvm::all_of(op->getOperands().slice(2, 4),
                    [&](Value value) { return hasShape(value, {rank}); }) ||
      !hasShape(op.getOperand(6), {rank, 2}) ||
      !isReducerIdentity(reducer.getBody().front(), op.getOperand(1))) {
    return std::nullopt;
  }
  auto isConstant = [](std::optional<ShapeElement> element, int64_t value) {
    return element && element->constant == value;
  };
  for (unsigned operand : {3, 4, 5}) {
    for (int64_t i = 0; i < rank; ++i) {
      if (!isConstant(getShapeElement(op.getOperand(operand), {i}), 1)) {
        return std::nullopt;
      }
    }
  }
  std::optional<int64_t> axis;
  for (int64_t i = 0; i < rank; ++i) {
    std::optional<ShapeElement> window = getShapeElement(op.getOperand(2), {i});
    if (isConstant(window, 1)) {
      continue;
    }
    if (axis || !window || !isDimSize(*window, input, i)) {
      return std::nullopt;
    }
    axis = i;
  }
  if (!axis) {
    return std::nullopt;
  }
  Value padding = op.getOperand(6);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != *axis && !(isConstant(getShapeElement(padding, {i, 0}), 0) &&
                        isConstant(getShapeElement(padding, {i, 1}), 0))) {
      return std::nullopt;
    }
  }
  std::optional<ShapeElement> low = getShapeElement(padding, {*axis, 0});
  std::optional<ShapeElement> high = getShapeElement(padding, {*axis, 1});
  if (!low || !high) {
    return std::nullopt;
  }
  if (isDimSizeMinusOne(*low, input, *axis) && isConstant(high, 0)) {
    return DynamicReduceWindowScan{*axis, /*isPostfix=*/false, reducer};
  }
  if (isConstant(low, 0) && isDimSizeMinusOne(*high, input, *axis)) {
    return DynamicReduceWindowScan{*axis, /*isPostfix=*/true, reducer};
  }
  return std::nullopt;
}

struct DynamicReduceWindowScanConversion final
    : OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<DynamicReduceWindowScan> scan =
        matchDynamicReduceWindowScan(op);
    if (!scan) {
      return rewriter.notifyMatchFailure(op, "not a scan window");
    }
    Value result = createReduceWindowScan(
        rewriter, op.getLoc(), op.getOperand(0), op.getOperand(1), scan->axis,
        scan->isPostfix, [&](Region &region) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.cloneRegionBefore(scan->reducer.getBody(), region,
                                     region.end());
          Operation *terminator = region.front().getTerminator();
          rewriter.setInsertionPoint(terminator);
          rewriter.replaceOpWithNewOp<mlir::stablehlo::ReturnOp>(
              terminator, terminator->getOperands());
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

struct TopkOpConversion final : OpConversionPattern<chlo::TopKOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(chlo::TopKOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();

    auto inputValuesType = dyn_cast<ShapedType>(operand.getType());
    auto outputValuesType = dyn_cast<ShapedType>(op.getValues().getType());
    auto outputIndicesType = dyn_cast<ShapedType>(op.getIndices().getType());
    if (!inputValuesType || !outputValuesType || !outputIndicesType) {
      return rewriter.notifyMatchFailure(
          op, "Input and output must be of ShapedType");
    }

    Type valueElementType = inputValuesType.getElementType();
    Type indicesElementType = outputIndicesType.getElementType();
    // Only handle integer types for indices. Index type is not supported.
    if (!isa<IntegerType>(indicesElementType)) {
      return rewriter.notifyMatchFailure(
          op, "Output indices must be of integer type.");
    }

    // Create and initialize output tensors for LinalgExt TopK results
    // Define the output types based on the results of CHLO TopK
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getOperand());
    mixedSizes.back() = rewriter.getIndexAttr(adaptor.getK());
    Value emptyTensorOutputValues = mlir::tensor::EmptyOp::create(
        rewriter, loc, mixedSizes, valueElementType);
    Value emptyTensorOutputIndices = mlir::tensor::EmptyOp::create(
        rewriter, loc, mixedSizes, indicesElementType);
    // Initialize indices to 0 and values to negative infinity
    TypedAttr negInfAttr;
    if (auto intType = dyn_cast<IntegerType>(valueElementType)) {
      negInfAttr = rewriter.getIntegerAttr(
          intType, APInt::getSignedMinValue(intType.getWidth()));
    } else {
      auto negApFloat =
          APFloat::getInf(cast<FloatType>(valueElementType).getFloatSemantics(),
                          /*Negative=*/true);
      negInfAttr = rewriter.getFloatAttr(valueElementType, negApFloat);
    }
    Value negInf = arith::ConstantOp::create(rewriter, loc, negInfAttr);
    TypedAttr posInfAttr = rewriter.getIntegerAttr(
        indicesElementType, APInt::getSignedMaxValue(32));
    Value posInf = arith::ConstantOp::create(rewriter, loc, posInfAttr);
    Value negInfTensor =
        linalg::FillOp::create(rewriter, loc, negInf, emptyTensorOutputValues)
            .result();
    Value posInfTensor =
        linalg::FillOp::create(rewriter, loc, posInf, emptyTensorOutputIndices)
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
        op, newResultTypes, /*values=*/operand, /*indices=*/Value(),
        /*output_values=*/negInfTensor, /*output_indices=*/posInfTensor, kDim);

    // Define the region of TopK with a GT comparison
    llvm::Repeated<Type> types(2, valueElementType);
    SmallVector<Location> locations(2, loc);
    Block *block =
        rewriter.createBlock(&topkOp.getRegion(), {}, types, locations);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      Value lhs = block->getArgument(0);
      Value rhs = block->getArgument(1);
      Value condition;
      if (isa<IntegerType>(valueElementType)) {
        condition = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::sge, lhs, rhs);
      } else {
        condition = arith::CmpFOp::create(rewriter, loc,
                                          arith::CmpFPredicate::OGT, lhs, rhs);
      }
      IREE::LinalgExt::YieldOp::create(rewriter, loc, condition);
    }

    return success();
  }
};

// Converts StableHLO ops on 0-d tensors inside LinalgExt op regions.
static void populateLinalgExtRegionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
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
// A module pass: the custom call names its reducer by symbol, so lowering it
// reads a sibling function.
struct LegalizeDynamicReduceWindow final
    : impl::LegalizeDynamicReduceWindowBase<LegalizeDynamicReduceWindow> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect, linalg::LinalgDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect,
                complex::ComplexDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    StableHloToStdTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    patterns.add<DynamicReduceWindowScanConversion>(typeConverter, context);
    populateLinalgExtRegionPatterns(context, typeConverter, &patterns);

    // Bodies of LinalgExt ops created earlier are left to the general
    // lowering, which handles ops the scan's body patterns do not.
    llvm::DenseSet<Operation *> earlierOps;
    getOperation().walk([&](Operation *op) {
      if (isa<IREE::LinalgExt::IREELinalgExtDialect>(op->getDialect())) {
        earlierOps.insert(op);
      }
    });
    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalDialect<mlir::stablehlo::StablehloDialect>(
        [&](Operation *op) {
          return !isInBodyOfLinalgExtOps(op) ||
                 earlierOps.contains(op->getParentRegion()->getParentOp());
        });
    target.addDynamicallyLegalOp<mlir::stablehlo::CustomCallOp>(
        [](mlir::stablehlo::CustomCallOp op) {
          return !matchDynamicReduceWindowScan(op);
        });
    target.addLegalOp<UnrealizedConversionCastOp>();
    // JAX emits the reducers public, so symbol DCE would keep them.
    llvm::SetVector<func::FuncOp> reducers;
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      if (std::optional<DynamicReduceWindowScan> scan =
              matchDynamicReduceWindowScan(op)) {
        reducers.insert(scan->reducer);
      }
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
    for (func::FuncOp reducer : reducers) {
      if (SymbolTable::symbolKnownUseEmpty(reducer, getOperation())) {
        reducer.erase();
      }
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

  populateLinalgExtRegionPatterns(context, typeConverter, patterns);
}

} // namespace mlir::iree_compiler::stablehlo
