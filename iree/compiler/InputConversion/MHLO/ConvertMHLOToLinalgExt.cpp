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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Type convertIntegerToSignless(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

static Optional<Type> convertRank0TensorToScalar(RankedTensorType tensorType) {
  if (tensorType.getRank() != 0) return llvm::None;
  Type elementType = tensorType.getElementType();
  if (auto intType = elementType.dyn_cast<IntegerType>()) {
    elementType = convertIntegerToSignless(intType);
  }
  return elementType;
}

static Optional<Value> materializeCastToSignless(OpBuilder &builder,
                                                 IntegerType toType,
                                                 ValueRange inputs,
                                                 Location loc) {
  assert(inputs.size() == 1 && "too many inputs to type conversion");
  Value fromValue = inputs[0];
  auto fromType = fromValue.getType();
  if (fromType.isSignlessInteger() || !toType.isSignlessInteger())
    return llvm::None;
  // Use unrealized conversion casts to do signful->signless conversion.
  return builder.create<UnrealizedConversionCastOp>(loc, toType, fromValue)
      ->getResult(0);
}

static Optional<Value> materializeCastToScalar(OpBuilder &builder, Type toType,
                                               ValueRange inputs,
                                               Location loc) {
  assert(inputs.size() == 1 && "too many inputs to type conversion");
  Value fromValue = inputs[0];
  auto fromType = fromValue.getType().dyn_cast<RankedTensorType>();
  if (!fromType || fromType.getRank() != 0) return llvm::None;

  if (auto intFromType = fromType.getElementType().dyn_cast<IntegerType>()) {
    if (!intFromType.isSignlessInteger()) {
      if (!toType.isSignlessInteger()) return llvm::None;
      fromType = fromType.clone(toType).cast<RankedTensorType>();
      fromValue =
          builder.create<UnrealizedConversionCastOp>(loc, fromType, fromValue)
              ->getResult(0);
    }
  }

  Type extractType = fromType.getElementType();
  return builder.createOrFold<tensor::ExtractOp>(loc, extractType, fromValue);
}

/// Note: only designed to work for casts involving rank-0 tensors and scalars
/// implicitly captured within op regions.
class MhloToStdTypeConverter : public TypeConverter {
 public:
  MhloToStdTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion(convertRank0TensorToScalar);
    addConversion(convertIntegerToSignless);

    addArgumentMaterialization(materializeCastToScalar);
    addTargetMaterialization(materializeCastToScalar);
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

static SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t> ret;
  for (const APInt &element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
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
    Value result = lmhlo::HloOpToStdScalarOp::map<OpTy>(
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
    auto sortOp = rewriter.create<IREE::LinalgExt::SortOp>(
        mhloSortOp.getLoc(), mhloSortOp.getResultTypes(),
        /*inputs=*/ValueRange{}, adaptor.getOperands(),
        mhloSortOp.dimensionAttr());
    rewriter.inlineRegionBefore(mhloSortOp.comparator(), sortOp.region(),
                                sortOp.region().begin());
    Region &region = sortOp.region();
    Block &block = region.front();
    TypeConverter::SignatureConversion signature_converter(
        block.getNumArguments());
    for (auto en : llvm::enumerate(block.getArguments())) {
      signature_converter.addInputs(en.index(),
                                    getElementTypeOrSelf(en.value().getType()));
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
  ///
  /// TODO(hanchung): Add a pattern for legalizing mhlo.scatter to canonical
  /// form to MHLOToMHLOPreprocessingPass.
  static bool hasCanonicalDimensionNumbers(mhlo::ScatterOp op) {
    auto dimNumbers = op.scatter_dimension_numbers();
    auto indicesType = op.scatter_indices().getType().cast<ShapedType>();
    auto indicesRank = indicesType.getRank();

    if (indicesRank < 2) return false;
    if (dimNumbers.getIndexVectorDim() != indicesRank - 1) return false;

    auto indexDepth = indicesType.getShape().back();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    if (scatterDimsToOperandDims.size() != indexDepth) return false;
    for (auto en : llvm::enumerate(scatterDimsToOperandDims)) {
      if (en.index() != en.value()) return false;
    }

    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    for (auto en : llvm::enumerate(insertedWindowDims)) {
      if (en.index() != en.value()) return false;
    }

    for (auto en : llvm::enumerate(dimNumbers.getUpdateWindowDims())) {
      if (en.index() + insertedWindowDims.size() != en.value()) return false;
    }

    return true;
  }

  static SmallVector<int64_t> getTiedResultOperandIndices(ValueRange operands) {
    // Mark linalg_ext.scatter::orinigal as readwrite tensor.
    return {0};
  }

  static LogicalResult collapseBatchDimsIfNeeded(Value &indices, Value &updates,
                                                 ImplicitLocOpBuilder &b) {
    auto indicesType = indices.getType().cast<ShapedType>();
    auto updatesType = updates.getType().cast<ShapedType>();
    if (indicesType.getRank() == 2) return success();

    int64_t batchSize = 1;
    auto indicesRank = indicesType.getRank();
    auto shape = indicesType.getShape();
    for (auto i : shape.drop_back(1)) {  // drop index_detph_dim
      if (i == ShapedType::kDynamicSize) {
        batchSize = ShapedType::kDynamicSize;
      } else if (batchSize != ShapedType::kDynamicSize) {
        batchSize *= i;
      }
    }

    SmallVector<ReassociationIndices> map;
    map.emplace_back(
        llvm::to_vector<4>(llvm::seq<int64_t>(0, indicesRank - 1)));
    map.emplace_back(1, indicesRank - 1);
    auto resultType = RankedTensorType::get({batchSize, shape.back()},
                                            indicesType.getElementType());
    indices = b.create<linalg::TensorCollapseShapeOp>(resultType, indices, map);

    auto updateShape = updatesType.getShape().drop_front(shape.size() - 1);
    SmallVector<int64_t> collapsedUpdateShape = {batchSize};
    collapsedUpdateShape.append(updateShape.begin(), updateShape.end());
    resultType = RankedTensorType::get(collapsedUpdateShape,
                                       updatesType.getElementType());
    // The batching dims are identical.
    map.pop_back();
    for (auto i : llvm::seq<int64_t>(indicesRank - 1, updatesType.getRank())) {
      map.emplace_back(1, i);
    }
    updates = b.create<linalg::TensorCollapseShapeOp>(resultType, updates, map);

    return success();
  }

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!hasCanonicalDimensionNumbers(op)) return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value original = adaptor.operand();
    Value indices = adaptor.scatter_indices();
    Value updates = adaptor.updates();

    if (failed(collapseBatchDimsIfNeeded(indices, updates, b))) {
      return failure();
    }
    auto scatterOp = rewriter.create<IREE::LinalgExt::ScatterOp>(
        op.getLoc(), op->getResultTypes(), ValueRange{updates, indices},
        ValueRange{original});

    rewriter.inlineRegionBefore(op.update_computation(), scatterOp.region(),
                                scatterOp.region().begin());
    Region &region = scatterOp.region();
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

    SmallVector<Value> dynSizes;
    for (auto en : llvm::enumerate(realType.getShape())) {
      if (en.value() == ShapedType::kDynamicSize) {
        dynSizes.push_back(b.create<tensor::DimOp>(real, en.index()));
      }
    }
    Value initTensor = b.create<linalg::InitTensorOp>(
        dynSizes, realType.getShape(), realType.getElementType());

    SmallVector<AffineMap> maps;
    maps.push_back(
        AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext()));
    maps.push_back(b.getMultiDimIdentityMap(rank));
    SmallVector<StringRef> iterTypes(rank, getParallelIteratorTypeName());

    Value indices = getBitReversalBuffer(b, fftLength);
    auto genericOp = b.create<linalg::GenericOp>(
        TypeRange{realType}, indices, initTensor, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          SmallVector<Value> ivs;
          for (auto i : llvm::seq<unsigned>(0, rank - 1)) {
            ivs.push_back(b.create<linalg::IndexOp>(loc, i));
          }
          ivs.push_back(
              b.create<arith::IndexCastOp>(loc, args[0], b.getIndexType()));
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
    auto operandType = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !operandType.hasStaticShape()) {
      return failure();
    }
    int fftLength = op.fft_length().getSplatValue<IntegerAttr>().getInt();
    if (fftLength & (fftLength - 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected FFT length to be a power of two");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> results =
        getBitReversalOrder(b, adaptor.operand(), fftLength);
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
    SmallVector<OpFoldResult> sizes;
    Value operand = adaptor.operand();
    for (auto dim : llvm::enumerate(operandType.getShape().drop_back())) {
      if (dim.value() != ShapedType::kDynamicSize) {
        sizes.push_back(b.getIndexAttr(dim.value()));
      } else {
        sizes.push_back(b.createOrFold<tensor::DimOp>(operand, dim.index()));
      }
    }
    sizes.push_back(b.getIndexAttr(shape.back()));
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
    SmallVector<Value> dynSizes;
    for (auto en : llvm::enumerate(ty.getShape())) {
      if (en.value() == ShapedType::kDynamicSize) {
        dynSizes.push_back(rewriter.create<tensor::DimOp>(
            loc, adaptor.getOperands()[0], en.index()));
      }
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, dynSizes, ty.getShape(), ty.getElementType());
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::ReverseOp>(
        op, op->getResultTypes(), adaptor.getOperands(), initTensor,
        op.dimensions());
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
                IREE::Flow::FlowDialect, StandardOpsDialect,
                mlir::math::MathDialect, mlir::arith::ArithmeticDialect,
                complex::ComplexDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    MLIRContext *context = &getContext();

    MhloToStdTypeConverter typeConverter;
    patterns.insert<SortOpConversion, ScatterOpConversion, FftOpConversion,
                    ReverseOpConversion>(typeConverter, context);
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
                LinalgExtRegionHLOOpConversion<mhlo::CosOp>,
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
                LinalgExtRegionHLOOpConversion<mhlo::SinOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SqrtOp>,
                LinalgExtRegionHLOOpConversion<mhlo::SubOp>,
                LinalgExtRegionHLOOpConversion<mhlo::TanhOp>,
                LinalgExtRegionHLOOpConversion<mhlo::XorOp>,
                LinalgExtRegionReturnOpConversion>(typeConverter, context);

    ConversionTarget target(getContext());
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect,
                           linalg::LinalgDialect, IREE::Flow::FlowDialect,
                           StandardOpsDialect, mlir::math::MathDialect,
                           mlir::arith::ArithmeticDialect,
                           tensor::TensorDialect, complex::ComplexDialect>();
    target.addIllegalOp<mhlo::SortOp, mhlo::ScatterOp, mhlo::FftOp,
                        mhlo::ReverseOp>();
    // FFT conversion creates complex ops which will be converted by the normal
    // MHLO lowering, but these should still be converted if present inside
    // other linalg_ext op regions.
    target.addDynamicallyLegalOp<mhlo::ComplexOp>(
        [](mhlo::ComplexOp complexOp) {
          return !isInBodyOfLinalgExtOps(complexOp);
        });
    // We deliberately allow unrealized casts to persist. These should fall away
    // when the rest of MHLO is converted.
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertMHLOToLinalgExtPass() {
  return std::make_unique<ConvertMHLOToLinalgExtPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
