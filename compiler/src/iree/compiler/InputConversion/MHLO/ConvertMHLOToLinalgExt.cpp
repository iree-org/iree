// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <complex>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
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

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {

static Type convertIntegerToSignless(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

static std::optional<Type> convertRank0TensorToScalar(
    RankedTensorType tensorType) {
  if (tensorType.getRank() != 0) return std::nullopt;
  Type elementType = tensorType.getElementType();
  if (auto intType = elementType.dyn_cast<IntegerType>()) {
    elementType = convertIntegerToSignless(intType);
  }
  return elementType;
}

static Type convertShapedToSignless(ShapedType shapedType) {
  if (auto intType = shapedType.getElementType().dyn_cast<IntegerType>())
    return shapedType.clone(convertIntegerToSignless(intType));
  return shapedType;
}

static std::optional<Value> materializeCast(OpBuilder &builder, Type toType,
                                            ValueRange inputs, Location loc) {
  assert(inputs.size() == 1 && "too many inputs to type conversion");
  Value fromValue = inputs[0];
  auto fromType = fromValue.getType().dyn_cast<RankedTensorType>();
  if (!fromType) return std::nullopt;

  if (auto intFromType = fromType.getElementType().dyn_cast<IntegerType>()) {
    Type castType = getElementTypeOrSelf(toType);
    if (auto shapedType = fromType.dyn_cast<ShapedType>())
      castType = shapedType.clone(castType);

    if (castType != fromType)
      fromValue = builder.create<tensor::BitcastOp>(loc, castType, fromValue)
                      ->getResult(0);
  }

  if (fromType.getRank() != 0) return fromValue;

  Type extractType = getElementTypeOrSelf(toType);
  return builder.createOrFold<tensor::ExtractOp>(loc, extractType, fromValue);
}

/// Note: only designed to work for casts involving rank-0 tensors and scalars
/// implicitly captured within op regions.
class MhloToStdTypeConverter : public TypeConverter {
 public:
  MhloToStdTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion(convertShapedToSignless);
    addConversion(convertRank0TensorToScalar);
    addConversion(convertIntegerToSignless);

    addArgumentMaterialization(materializeCast);
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static bool isInBodyOfLinalgExtOps(Operation *op) {
  auto parent_op = op->getParentRegion()->getParentOp();
  return parent_op->getDialect() ==
         parent_op->getContext()
             ->getLoadedDialect<IREE::LinalgExt::IREELinalgExtDialect>();
}

//===----------------------------------------------------------------------===//
// Region operations lowering.
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct LinalgExtRegionHLOOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isInBodyOfLinalgExtOps(op)) return failure();
    TensorType origRetType = op.getType().template dyn_cast<TensorType>();
    if (!origRetType) return failure();
    SmallVector<Value> scalarArgs;
    Type newRetType = getElementTypeOrSelf(
        this->typeConverter->convertType(origRetType.getElementType()));
    Value result = mhlo::MhloOpToStdScalarOp::mapOp<OpTy>(
        op, newRetType, adaptor.getOperands(), &rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinalgExtRegionReturnOpConversion
    : public OpConversionPattern<mhlo::ReturnOp> {
  using OpConversionPattern<mhlo::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!isInBodyOfLinalgExtOps(op)) return failure();
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::YieldOp>(
        op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

struct SortOpConversion : public OpConversionPattern<mhlo::SortOp> {
  using OpConversionPattern<mhlo::SortOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SortOp mhloSortOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = mhloSortOp.getLoc();

    llvm::SmallVector<Type> resultTypes;
    if (this->typeConverter
            ->convertTypes(mhloSortOp.getResultTypes(), resultTypes)
            .failed()) {
      return failure();
    };
    auto sortOp = rewriter.create<IREE::LinalgExt::SortOp>(
        loc, resultTypes,
        /*inputs=*/ValueRange{}, adaptor.getOperands(),
        mhloSortOp.getDimensionAttr());
    rewriter.inlineRegionBefore(mhloSortOp.getComparator(), sortOp.getRegion(),
                                sortOp.getRegion().begin());
    Region &region = sortOp.getRegion();
    Block &block = region.front();
    TypeConverter::SignatureConversion signature_converter(
        block.getNumArguments());
    for (auto en : llvm::enumerate(block.getArguments())) {
      signature_converter.addInputs(
          en.index(), this->typeConverter->convertType(
                          getElementTypeOrSelf(en.value().getType())));
    }
    rewriter.applySignatureConversion(&region, signature_converter);

    rewriter.replaceOp(mhloSortOp, sortOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

struct ScatterOpConversion : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  /// Returns true if the `dimensionNumbers` from the mhlo.scatter op follows a
  /// canonical form:
  ///
  /// * The rank of indices is greater than or equal to two.
  /// * The index_vector_dim is the last dim of indices.
  /// * Scatter dims to operand dims order: (0, ... , n)
  /// * Inserted window dims order: (0, ... , d)
  /// * Update window dims order: (d + 1, ... , m)
  static bool hasCanonicalDimensionNumbers(mhlo::ScatterOp op) {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indicesType = op.getScatterIndices().getType().cast<ShapedType>();
    auto indicesRank = indicesType.getRank();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indexDepth = indicesType.getShape().back();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();

    if (indicesRank != 2) return false;
    if (indexVectorDim != indicesRank - 1) return false;
    if (scatterDimsToOperandDims.size() != indexDepth) return false;

    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    for (auto en : llvm::enumerate(insertedWindowDims)) {
      if (en.index() != en.value()) return false;
    }

    // Check that there is only one batch dimension in the updates.
    for (auto en : llvm::enumerate(dimNumbers.getUpdateWindowDims())) {
      if (en.index() + 1 != en.value()) return false;
    }

    return true;
  }

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!hasCanonicalDimensionNumbers(op)) return failure();
    if (llvm::size(op.getInputs()) != 1)
      return op.emitError("NYI variadic operands scatter");
    if (llvm::size(op.getUpdates()) != 1)
      return op.emitError("NYI variadic updates scatter");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value original = adaptor.getInputs().front();
    Value indices = adaptor.getScatterIndices();
    Value updates = adaptor.getUpdates().front();

    llvm::SmallVector<int64_t> scatterDimMap;
    for (auto dim :
         op.getScatterDimensionNumbers().getScatterDimsToOperandDims()) {
      scatterDimMap.push_back(dim);
    }

    auto scatterOp = rewriter.create<IREE::LinalgExt::ScatterOp>(
        op.getLoc(), op->getResultTypes(), ValueRange{updates, indices},
        ValueRange{original}, scatterDimMap, op.getUniqueIndices());

    rewriter.inlineRegionBefore(op.getUpdateComputation(),
                                scatterOp.getRegion(),
                                scatterOp.getRegion().begin());
    Region &region = scatterOp.getRegion();
    TypeConverter::SignatureConversion signatureConverter(2);
    Type argType = getElementTypeOrSelf(original.getType());
    // mhlo.scatter ops takes:
    //   output[O] = update_computation(output[O], updates[U])
    // where output[O] maps to block args #1 in linalg_ext.scatter ops.
    signatureConverter.addInputs(1, argType);
    signatureConverter.addInputs(0, argType);
    rewriter.applySignatureConversion(&region, signatureConverter);

    rewriter.replaceOp(op, scatterOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

struct FftOpConversion : public OpConversionPattern<mhlo::FftOp> {
  using OpConversionPattern<mhlo::FftOp>::OpConversionPattern;

  static Value getBitReversalBuffer(ImplicitLocOpBuilder &b, int fftLength) {
    SmallVector<Attribute> values;
    int logn = std::log(fftLength) / std::log(2);
    for (int i = 0; i < fftLength; ++i) {
      int r = 0;
      for (int j = 0; j < logn; ++j) {
        r |= ((i >> j) & 1) << (logn - j - 1);
      }
      values.push_back(b.getI32IntegerAttr(r));
    }
    auto type = RankedTensorType::get({fftLength}, b.getI32Type());
    return b.create<arith::ConstantOp>(type,
                                       DenseIntElementsAttr::get(type, values));
  }

  static SmallVector<Value> getBitReversalOrder(ImplicitLocOpBuilder &b,
                                                Value real, int fftLength) {
    auto realType = real.getType().cast<ShapedType>();
    auto rank = realType.getRank();

    SmallVector<OpFoldResult> mixedSizes =
        tensor::createDimValues(b, b.getLoc(), real);
    Value emptyTensor =
        b.create<tensor::EmptyOp>(mixedSizes, realType.getElementType());

    SmallVector<AffineMap> maps;
    maps.push_back(
        AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext()));
    maps.push_back(b.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);

    Value indices = getBitReversalBuffer(b, fftLength);
    auto genericOp = b.create<linalg::GenericOp>(
        TypeRange{realType}, indices, emptyTensor, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          SmallVector<Value> ivs;
          for (auto i : llvm::seq<unsigned>(0, rank - 1)) {
            ivs.push_back(b.create<linalg::IndexOp>(loc, i));
          }
          ivs.push_back(
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), args[0]));
          b.create<linalg::YieldOp>(
              loc, b.create<tensor::ExtractOp>(loc, real, ivs).getResult());
        });
    return {
        genericOp.getResult(0),
        b.create<arith::ConstantOp>(
            realType, DenseFPElementsAttr::get(
                          realType, b.getF32FloatAttr(0.0).cast<Attribute>()))};
  }

  static SmallVector<Value> getCoeffConstants(ImplicitLocOpBuilder &b,
                                              int stage) {
    constexpr std::complex<double> kI(0, 1);
    int m = 1 << stage;
    int mh = m >> 1;
    SmallVector<Attribute> real, imag;
    for (auto i : llvm::seq<unsigned>(0, mh)) {
      auto v = std::exp(-2 * M_PI * i / m * kI);
      real.push_back(b.getF32FloatAttr(v.real()));
      imag.push_back(b.getF32FloatAttr(v.imag()));
    }
    auto type = RankedTensorType::get({mh}, b.getF32Type());
    return {
        b.create<arith::ConstantOp>(type, DenseFPElementsAttr::get(type, real)),
        b.create<arith::ConstantOp>(type,
                                    DenseFPElementsAttr::get(type, imag))};
  }

  LogicalResult matchAndRewrite(
      mhlo::FftOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Only handle 2^n fft length.
    auto operandType =
        adaptor.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !operandType.hasStaticShape()) {
      return failure();
    }
    int fftLength = op.getFftLength().getSplatValue<IntegerAttr>().getInt();
    if (fftLength & (fftLength - 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected FFT length to be a power of two");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> results =
        getBitReversalOrder(b, adaptor.getOperand(), fftLength);
    int lognPlus1 = std::log(fftLength) / std::log(2) + 1;
    for (auto s : llvm::seq<unsigned>(1, lognPlus1)) {
      SmallVector<Value> inputs;
      inputs.push_back(b.create<arith::ConstantIndexOp>(s));
      inputs.append(getCoeffConstants(b, s));
      auto fft = b.create<IREE::LinalgExt::FftOp>(
          TypeRange{results[0].getType(), results[1].getType()}, inputs,
          results);
      results = fft.getResults();
    }

    SmallVector<int64_t> shape(operandType.getShape().begin(),
                               operandType.getShape().end());
    shape.back() = fftLength / 2 + 1;
    auto ty = RankedTensorType::get(shape, operandType.getElementType());
    SmallVector<OpFoldResult> offsets(ty.getRank(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(ty.getRank(), b.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        tensor::createDimValues(b, b.getLoc(), adaptor.getOperand());
    sizes.back() = b.getIndexAttr(shape.back());
    auto real = b.create<tensor::ExtractSliceOp>(ty, results[0], offsets, sizes,
                                                 strides);
    auto imag = b.create<tensor::ExtractSliceOp>(ty, results[1], offsets, sizes,
                                                 strides);
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, op.getType(), real, imag);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

struct ReverseOpConversion : public OpConversionPattern<mhlo::ReverseOp> {
  using OpConversionPattern<mhlo::ReverseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReverseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto ty = adaptor.getOperands()[0].getType().dyn_cast<RankedTensorType>();
    if (!ty) return failure();

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> mixedSizes =
        tensor::createDimValues(rewriter, loc, adaptor.getOperands()[0]);
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, mixedSizes, ty.getElementType());
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::ReverseOp>(
        op, op->getResultTypes(), adaptor.getOperands(), emptyTensor,
        op.getDimensions());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

struct TopkOpConversion : public OpConversionPattern<chlo::TopKOp> {
  using OpConversionPattern<chlo::TopKOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      chlo::TopKOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();

    auto inputValuesType = operand.getType().dyn_cast<ShapedType>();
    auto outputValuesType = op.getValues().getType().dyn_cast<ShapedType>();
    auto outputIndicesType = op.getIndices().getType().dyn_cast<ShapedType>();
    if (!inputValuesType || !outputValuesType || !outputIndicesType) {
      return rewriter.notifyMatchFailure(
          op, "Input and output must be of ShapedType");
    }

    Type valueElementType = outputValuesType.getElementType();
    Type indicesElementType = outputIndicesType.getElementType();
    // Only handle integer types for indicies. Index type is not supported.
    if (!indicesElementType.isa<IntegerType>()) {
      return rewriter.notifyMatchFailure(
          op, "Output indices must be of integer type.");
    }

    // Create and initialize output tensors for LinalgExt TopK results
    // Define the output types based on the results of CHLO TopK
    SmallVector<OpFoldResult> mixedSizes =
        tensor::createDimValues(rewriter, loc, adaptor.getOperand());
    mixedSizes.back() = rewriter.getIndexAttr(adaptor.getK());
    Value emptyTensorOutputValues = rewriter.create<mlir::tensor::EmptyOp>(
        loc, mixedSizes, valueElementType);
    Value emptyTensorOutputIndices = rewriter.create<mlir::tensor::EmptyOp>(
        loc, mixedSizes, indicesElementType);
    // Initialize indices to 0 and values to negative infinity
    TypedAttr negInfAttr;
    if (auto intType = valueElementType.dyn_cast<IntegerType>()) {
      negInfAttr = rewriter.getIntegerAttr(
          intType, APInt::getSignedMinValue(intType.getWidth()));
    } else {
      auto negApFloat = APFloat::getInf(
          valueElementType.cast<FloatType>().getFloatSemantics(),
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
    auto topkOp = rewriter.replaceOpWithNewOp<IREE::LinalgExt::TopkOp>(
        op, op->getResultTypes(), ValueRange{operand},
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
      if (valueElementType.isa<IntegerType>()) {
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

struct ConvertMHLOToLinalgExtPass
    : public ConvertMHLOToLinalgExtBase<ConvertMHLOToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect, linalg::LinalgDialect,
                IREE::Flow::FlowDialect, mlir::cf::ControlFlowDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect,
                complex::ComplexDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    MLIRContext *context = &getContext();

    MhloToStdTypeConverter typeConverter;
    patterns.insert<SortOpConversion, ScatterOpConversion, FftOpConversion,
                    ReverseOpConversion, TopkOpConversion>(typeConverter,
                                                           context);
    // FIXME: It shouldn't be necessary to list every matching MHLO op here,
    // especially since they're already listed in
    // populateHLOToLinalgConversionPattern and in HloOpToStdScalarOp. These
    // lists are all the same. Can we leverage SFINAE here?
    patterns
        .insert<LinalgExtRegionHLOOpConversion<mhlo::AbsOp>,
                LinalgExtRegionHLOOpConversion<mhlo::AddOp>,
                LinalgExtRegionHLOOpConversion<mhlo::AndOp>,
                LinalgExtRegionHLOOpConversion<mhlo::Atan2Op>,
                LinalgExtRegionHLOOpConversion<mhlo::BitcastConvertOp>,
                LinalgExtRegionHLOOpConversion<mhlo::CeilOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ClampOp>,
                LinalgExtRegionHLOOpConversion<mhlo::CompareOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ComplexOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ConvertOp>,
                LinalgExtRegionHLOOpConversion<mhlo::CopyOp>,
                LinalgExtRegionHLOOpConversion<mhlo::CosineOp>,
                LinalgExtRegionHLOOpConversion<mhlo::DivOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ExpOp>,
                LinalgExtRegionHLOOpConversion<mhlo::Expm1Op>,
                LinalgExtRegionHLOOpConversion<mhlo::FloorOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ImagOp>,
                LinalgExtRegionHLOOpConversion<mhlo::IsFiniteOp>,
                LinalgExtRegionHLOOpConversion<mhlo::LogOp>,
                LinalgExtRegionHLOOpConversion<mhlo::LogisticOp>,
                LinalgExtRegionHLOOpConversion<mhlo::Log1pOp>,
                LinalgExtRegionHLOOpConversion<mhlo::MaxOp>,
                LinalgExtRegionHLOOpConversion<mhlo::MinOp>,
                LinalgExtRegionHLOOpConversion<mhlo::MulOp>,
                LinalgExtRegionHLOOpConversion<mhlo::NegOp>,
                LinalgExtRegionHLOOpConversion<mhlo::NotOp>,
                LinalgExtRegionHLOOpConversion<mhlo::OrOp>,
                LinalgExtRegionHLOOpConversion<mhlo::PowOp>,
                LinalgExtRegionHLOOpConversion<mhlo::RealOp>,
                LinalgExtRegionHLOOpConversion<mhlo::RemOp>,
                LinalgExtRegionHLOOpConversion<mhlo::RsqrtOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SelectOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ShiftLeftOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ShiftRightArithmeticOp>,
                LinalgExtRegionHLOOpConversion<mhlo::ShiftRightLogicalOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SignOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SineOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SqrtOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SubtractOp>,
                LinalgExtRegionHLOOpConversion<mhlo::TanhOp>,
                LinalgExtRegionHLOOpConversion<mhlo::XorOp>,
                LinalgExtRegionReturnOpConversion>(typeConverter, context);

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
    target.addIllegalOp<mhlo::SortOp, mhlo::FftOp, mhlo::ReverseOp>();
    // FFT conversion creates complex ops which will be converted by the normal
    // MHLO lowering, but these should still be converted if present inside
    // other linalg_ext op regions.
    target.addDynamicallyLegalOp<mhlo::ComplexOp>(
        [](mhlo::ComplexOp complexOp) {
          return !isInBodyOfLinalgExtOps(complexOp);
        });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertMHLOToLinalgExtPass() {
  return std::make_unique<ConvertMHLOToLinalgExtPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
