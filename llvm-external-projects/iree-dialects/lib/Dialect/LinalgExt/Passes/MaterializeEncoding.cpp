// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::LinalgExt;

//===---------------------------------------------------------------------===//
// Utility methods
//===---------------------------------------------------------------------===//

/// Extract encoding from the `tensorType` if specified.
static Optional<TensorEncoding> getEncoding(RankedTensorType tensorType) {
  auto encodingAttr = tensorType.getEncoding().dyn_cast_or_null<EncodingAttr>();
  if (!encodingAttr)
    return std::nullopt;
  return encodingAttr.getEncoding().getValue();
}

/// For a given tensor type with an encoding, return the materialized
/// type to use for it. If no encoding is set, then return the tensor type
/// itself.
static RankedTensorType
getMaterializedType(RankedTensorType tensorType,
                    MaterializeEncodingFn materializeEncodingFn) {
  Optional<TensorEncoding> encoding = getEncoding(tensorType);
  if (!encoding)
    return tensorType;
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);
  if (failed(materializeEncodingInfo)) {
    return tensorType;
  }
  return tensor::PackOp::inferPackedType(
             tensorType, materializeEncodingInfo->innerTileSizes,
             materializeEncodingInfo->innerDimsPos,
             materializeEncodingInfo->outerDimsPerm)
      .cast<RankedTensorType>();
}

//===---------------------------------------------------------------------===//
// Methods to convert the encoding to parameters of the Pack operation
//===---------------------------------------------------------------------===//

/// Given the `encoding` return the `MaterializeEncodingInfo` to use for
/// materializing the pack op.
// TODO(ravishankarm): This is currently hard-coded here for convenience. When
// used in IREE, this will be computed based on the architecture information in
// `hal.executable.variant`.
// A real implementation would return tile sizes that depend on at least the
// `tensorType`'s element type (e.g. different tile sizes for i8 vs f32, because
// the SIMD instructions may have different shapes).
// Moreover, in a real implementation, the tile sizes would typically also
// depend on target information. This is demonstrated in
// iree/compiler/src/iree/compiler/Codegen/Common/MaterializeEncodingPass.cpp
static FailureOr<MaterializeEncodingInfo>
chooseEncodingInfo(RankedTensorType tensorType) {
  Optional<TensorEncoding> encoding = getEncoding(tensorType);
  if (!encoding)
    return failure();
  switch (*encoding) {
  case TensorEncoding::MATMUL_F32F32F32_LHS:
  case TensorEncoding::MATMUL_I8I8I32_LHS:
    return MaterializeEncodingInfo{{0, 1}, {8, 4}, {}};
    break;
  case TensorEncoding::MATMUL_F32F32F32_RHS:
  case TensorEncoding::MATMUL_I8I8I32_RHS:
    return MaterializeEncodingInfo{{1, 0}, {8, 4}, {1, 0}};
    break;
  case TensorEncoding::MATMUL_F32F32F32_RESULT:
  case TensorEncoding::MATMUL_I8I8I32_RESULT:
    return MaterializeEncodingInfo{{0, 1}, {8, 8}, {}};
    break;
  default:
    return failure();
  }
}

//===---------------------------------------------------------------------===//
// Methods to convert `set_encoding` and `unset_encoding` operations
// to `pack` and `unpack` operations respectively.
//===---------------------------------------------------------------------===//

/// Utility method to get the optional padding value to use with pack operation
/// if source is defined using a `tensor.pad` operation. Note `source` is
/// passed by reference. It is updated to use the source of the pad operation.
static Optional<Value> getPaddingValue(Value &source) {
  auto padOp = source.getDefiningOp<tensor::PadOp>();
  if (!padOp || padOp.getNofold() || !padOp.hasZeroLowPad())
    return std::nullopt;

  Value constantPaddingValue = padOp.getConstantPaddingValue();
  if (!constantPaddingValue)
    return std::nullopt;

  source = padOp.getSource();
  return constantPaddingValue;
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// For now this takes a `paddingValue` as input. The source is also taken
/// as input so that these could be used with `OpConversionPatterns`.
static FailureOr<tensor::PackOp> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, SetEncodingOp encodingOp, Value source,
    MaterializeEncodingFn materializeEncodingFn,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  RankedTensorType resultType = encodingOp.getResultType();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(resultType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled result encoding");
  }
  // Create `tensor.empty` operation for the result of the pack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> sourceDims = getDims(rewriter, loc, source);
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, resultType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  Optional<TensorEncoding> encoding = getEncoding(resultType);
  if (!encoding)
    return failure();
  SmallVector<OpFoldResult> resultDims = tensor::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr,
      materializeEncodingInfo->innerDimsPos,
      materializeEncodingInfo->outerDimsPerm);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  resultType.getElementType());
  Optional<Value> paddingValue = getPaddingValue(source);
  auto packOp = rewriter.create<tensor::PackOp>(
      loc, source, emptyOp, materializeEncodingInfo->innerDimsPos,
      *innerTileSizesOfr, paddingValue, materializeEncodingInfo->outerDimsPerm);
  // As we rewrite the SetEncoding and its old result tensor, which used to hold
  // the TensorEncodingAttr, into a pack op with a new result tensor which does
  // not have a TensorEncodingAttr, we lose the information that used to be
  // stored in that attr. That shouldn't matter, as the purpose of that attr
  // was to enable exactly this rewrite, but there is a catch: at the moment,
  // in IREE's TileAndDistributeToWorkgroupsPass.cpp, we need the encoding value
  // again. See the comment there. So we re-add the attribute on the pack op
  // itself as a temporary work-around.
  packOp->setAttr(StringAttr::get(rewriter.getContext(), "encoding"),
                  EncodingAttr::get(rewriter.getContext(), *encoding));
  return packOp;
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// The source is taken as input so that these could be used with
/// `OpConversionPatterns`.
static FailureOr<tensor::UnPackOp> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, UnsetEncodingOp encodingOp, Value packedValue,
    MaterializeEncodingFn materializeEncodingFn,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  RankedTensorType sourceType = encodingOp.getSourceType();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(sourceType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled source encoding");
  }
  // Create an `tensor.empty` for the result of the unpack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> resultDims =
      getDims(rewriter, loc, encodingOp.getSource());
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  sourceType.getElementType());
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, sourceType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  return rewriter.create<tensor::UnPackOp>(
      loc, packedValue, emptyOp, materializeEncodingInfo->innerDimsPos,
      *innerTileSizesOfr, materializeEncodingInfo->outerDimsPerm);
}

/// Utility method to convert from `linalg.matmul` with
/// - lhs encoding of MATMUL_*_LHS
/// - rhs encoding of MATMUL_*_RHS
/// - result encoding of MATMUL_*_RESULT
/// to linalg.mmt4d op.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands, MaterializeEncodingFn,
                    MaterializeEncodingValueFn) {
  if (!matmulOp.hasTensorSemantics())
    return failure();
  auto inputs = matmulOp.getDpsInputOperands();
  auto outputs = matmulOp.getDpsInitOperands();
  Optional<TensorEncoding> lhsEncoding =
      getEncoding(inputs[0]->get().getType().cast<RankedTensorType>());
  Optional<TensorEncoding> rhsEncoding =
      getEncoding(inputs[1]->get().getType().cast<RankedTensorType>());
  Optional<TensorEncoding> resultEncoding =
      getEncoding(outputs[0]->get().getType().cast<RankedTensorType>());
  if (!lhsEncoding ||
      (lhsEncoding.value() != TensorEncoding::MATMUL_F32F32F32_LHS &&
       lhsEncoding.value() != TensorEncoding::MATMUL_I8I8I32_LHS) ||
      !rhsEncoding ||
      (rhsEncoding.value() != TensorEncoding::MATMUL_F32F32F32_RHS &&
       rhsEncoding.value() != TensorEncoding::MATMUL_I8I8I32_RHS) ||
      !resultEncoding ||
      (resultEncoding.value() != TensorEncoding::MATMUL_F32F32F32_RESULT &&
       resultEncoding.value() != TensorEncoding::MATMUL_I8I8I32_RESULT)) {
    return failure();
  }
  Operation *mmt4DOp = rewriter.create<linalg::Mmt4DOp>(
      matmulOp.getLoc(), convertedOutputOperands[0].getType(),
      convertedInputOperands, convertedOutputOperands);
  return mmt4DOp;
}

/// Utility method to convert from `linalg.fill` on `tensor` type with encoding
/// to fill of the materialized type
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::FillOp fillOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands, MaterializeEncodingFn,
                    MaterializeEncodingValueFn) {
  if (!fillOp.hasTensorSemantics())
    return failure();
  Operation *materializedFillOp = rewriter.create<linalg::FillOp>(
      fillOp.getLoc(), convertedOutputOperands[0].getType(),
      convertedInputOperands, convertedOutputOperands);
  return materializedFillOp;
}

/// Utility method to convert `tensor.empty` with encoding to a `tensor.empty`
/// of the materialized type.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, tensor::EmptyOp emptyOp,
                    ValueRange convertedOperands,
                    MaterializeEncodingFn materializeEncodingFn,
                    MaterializeEncodingValueFn materializeEncodingValueFn) {
  auto result = emptyOp.getResult();
  auto resultType = result.getType().cast<RankedTensorType>();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(resultType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(emptyOp, "unhandled result encoding");
  }
  Location loc = emptyOp.getLoc();
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, resultType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        emptyOp, "failed to generate runtime tile size query");
  }
  SmallVector<OpFoldResult> newShape = PackOp::getResultShape(
      rewriter, loc, emptyOp.getMixedSizes(), *innerTileSizesOfr,
      materializeEncodingInfo->innerDimsPos,
      materializeEncodingInfo->outerDimsPerm);
  Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
      loc, newShape, resultType.getElementType());
  return newEmptyOp;
}

namespace {
//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Convert `set_encoding` op to `pack` op.
struct SetEncodingOpToPackOpConversion
    : public OpMaterializeEncodingPattern<SetEncodingOp> {
  using OpMaterializeEncodingPattern<
      SetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(getTypeConverter())
            ->getMaterializeEncodingFn();
    // Pack op needs a padding value. Maybe that is an overkill. For now, just
    // use zero.
    auto packOp = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(packOp))
      return rewriter.notifyMatchFailure(encodingOp,
                                         "failed to convert to pack op");
    rewriter.replaceOp(encodingOp, packOp->getResult());
    return success();
  }
};

/// Convert `unset_encoding` op to `unpack` op.
struct UnsetEncodingOpToPackOpConversion
    : public OpMaterializeEncodingPattern<UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(UnsetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    auto unpackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(unpackOp))
      return rewriter.notifyMatchFailure(encodingOp,
                                         "failed to convert to unpack op");
    rewriter.replaceOp(encodingOp, unpackOp->getResult());
    return success();
  }
};

/// Generic pattern to convert operaiton that is in Destination Passing Style.
template <typename OpTy>
struct MaterializeDPSOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy dpsOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, dpsOp, adaptor.getInputs(), adaptor.getOutputs(),
        materializeEncodingFn, this->materializeEncodingValueFn);
    if (failed(convertedOp))
      return failure();
    rewriter.replaceOp(dpsOp, convertedOp.value()->getResults());
    return success();
  }
};

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, op, adaptor.getOperands(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(convertedOp))
      return failure();
    rewriter.replaceOp(op, convertedOp.value()->getResults());
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pass to materialize encoding
//===---------------------------------------------------------------------===//

struct MaterializeEncodingPass
    : public MaterializeEncodingBase<MaterializeEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void MaterializeEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    Operation *op = getOperation();
    RewritePatternSet patterns(context);
    MaterializeEncodingTypeConverter typeConverter(chooseEncodingInfo);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(patterns, target, typeConverter);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }

  // Add patterns to fold tensor.pack/unpack ops with tensor.pad/extract_slice
  // ops.
  {
    RewritePatternSet patterns(context);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
}
} // namespace

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    MaterializeEncodingFn materializeEncodingFn)
    : materializeEncodingFn(materializeEncodingFn) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion(
      [materializeEncodingFn](RankedTensorType t) -> RankedTensorType {
        return getMaterializedType(t, materializeEncodingFn);
      });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasEncoding = [](Type t) -> bool {
      auto tensorType = t.dyn_cast<RankedTensorType>();
      return tensorType && tensorType.getEncoding();
    };
    auto valueHasEncoding = [=](Value v) -> bool {
      return typeHasEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

void populateMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn) {

  // Add all patterns for converting from encoded type to the materialized type
  patterns.insert<MaterializeDPSOperation<linalg::FillOp>,
                  MaterializeDPSOperation<linalg::MatmulOp>,
                  MaterializeOperation<tensor::EmptyOp>,
                  SetEncodingOpToPackOpConversion,
                  UnsetEncodingOpToPackOpConversion>(
      patterns.getContext(), typeConverter, materializeEncodingValueFn);
  ::mlir::memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
}

std::unique_ptr<OperationPass<func::FuncOp>> createMaterializeEncodingPass() {
  return std::make_unique<MaterializeEncodingPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
