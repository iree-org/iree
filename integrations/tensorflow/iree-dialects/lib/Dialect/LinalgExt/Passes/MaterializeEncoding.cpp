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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::LinalgExt;

//===---------------------------------------------------------------------===//
// Methods to convert the encoding to parameters of the Pack operation
//===---------------------------------------------------------------------===//

/// Given the `encoding` return the `MaterializeEncodingInfo` to use for
/// materializing the pack op.
// TODO(ravishankarm): THis is currently hard-coded here for convenience. When
// used in IREE, this will be computed based on the architecture information in
// `hal.executable.variant`.
static FailureOr<MaterializeEncodingInfo>
getPackOpInfoFromEncoding(TensorEncoding encoding) {
  switch (encoding) {
  case TensorEncoding::GEMM_LHS:
    return MaterializeEncodingInfo{{0, 1}, {8, 4}, {}};
    break;
  case TensorEncoding::GEMM_RHS:
    return MaterializeEncodingInfo{{0, 1}, {4, 8}, {}};
    break;
  case TensorEncoding::GEMM_RESULT:
    return MaterializeEncodingInfo{{0, 1}, {8, 8}, {}};
    break;
  case TensorEncoding::GEMM_RHS_TRANSPOSE:
    return MaterializeEncodingInfo{{1, 0}, {8, 4}, {1, 0}};
    break;
  default:
    return failure();
  }
}

//===---------------------------------------------------------------------===//
// Utility methods
//===---------------------------------------------------------------------===//

/// Extract encoding from the `tensorType` if specified.
static Optional<TensorEncoding> getEncoding(RankedTensorType tensorType) {
  auto encodingAttr = tensorType.getEncoding().dyn_cast_or_null<EncodingAttr>();
  if (!encodingAttr)
    return llvm::None;
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
      materializeEncodingFn(encoding.value());
  if (failed(materializeEncodingInfo)) {
    return tensorType;
  }
  return PackOp::getPackedType(tensorType,
                               materializeEncodingInfo->innerTileSizes,
                               materializeEncodingInfo->innerDimsPos,
                               materializeEncodingInfo->outerDimsPerm)
      .cast<RankedTensorType>();
}

/// Helper methods to get `OpFoldResult` from `int64_t` values.
static OpFoldResult getAsOpFoldResult(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}
static SmallVector<OpFoldResult> getAsOpFoldResult(OpBuilder &builder,
                                                   ArrayRef<int64_t> values) {
  return llvm::to_vector(llvm::map_range(
      values, [&](int64_t v) { return getAsOpFoldResult(builder, v); }));
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
    return llvm::None;

  Value constantPaddingValue = padOp.getConstantPaddingValue();
  if (!constantPaddingValue)
    return llvm::None;

  source = padOp.getSource();
  return constantPaddingValue;
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// For now this takes a `paddingValue` as input. The source is also taken
/// as input so that these could be used with `OpConversionPatterns`.
static FailureOr<PackOp>
lowerSetEncodingOpToPackOp(RewriterBase &rewriter, SetEncodingOp encodingOp,
                           Value source,
                           MaterializeEncodingFn materializeEncodingFn) {
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(encodingOp.getResultTensorEncoding());
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled result encoding");
  }

  // Create `tensor.empty` operation for the result of the pack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> sourceDims = getDims(rewriter, loc, source);
  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter, materializeEncodingInfo->innerTileSizes);
  SmallVector<OpFoldResult> resultDims =
      PackOp::getResultShape(rewriter, loc, sourceDims, innerTileSizesOfr,
                             materializeEncodingInfo->innerDimsPos,
                             materializeEncodingInfo->outerDimsPerm);
  auto initTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultDims, encodingOp.getSourceType().getElementType());
  Optional<Value> paddingValue = getPaddingValue(source);
  return rewriter.create<PackOp>(
      loc, source, initTensor, materializeEncodingInfo->innerDimsPos,
      innerTileSizesOfr, paddingValue, materializeEncodingInfo->outerDimsPerm);
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// The source is taken as input so that these could be used with
/// `OpConversionPatterns`.
static FailureOr<UnPackOp>
lowerUnsetEncodingToUnpackOp(RewriterBase &rewriter, UnsetEncodingOp encodingOp,
                             Value packedValue,
                             MaterializeEncodingFn materializeEncodingFn) {
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(encodingOp.getSourceTensorEncoding());
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled source encoding");
  }
  // Create an `tensor.empty` for the result of the unpack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> resultDims =
      getDims(rewriter, loc, encodingOp.getSource());
  auto initTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultDims, encodingOp.getResultType().getElementType());

  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter, materializeEncodingInfo->innerTileSizes);
  return rewriter.create<UnPackOp>(
      loc, packedValue, initTensor, materializeEncodingInfo->innerDimsPos,
      innerTileSizesOfr, materializeEncodingInfo->outerDimsPerm);
}

/// Utility method to convert from `linalg.matmul` with
/// - lhs encoding of GEMM_LHS
/// - rhs encoding of GEMM_RHS_TRANSPOSE
/// - result encoding of GEMM_RESULT
/// to linalg.mmt4d op.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands,
                    MaterializeEncodingFn materializeEncodingFn) {
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
  if (!lhsEncoding || lhsEncoding.value() != TensorEncoding::GEMM_LHS ||
      !rhsEncoding ||
      rhsEncoding.value() != TensorEncoding::GEMM_RHS_TRANSPOSE ||
      !resultEncoding ||
      resultEncoding.value() != TensorEncoding::GEMM_RESULT) {
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
                    ValueRange convertedOutputOperands,
                    MaterializeEncodingFn materializeEncodingFn) {
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
                    MaterializeEncodingFn materializeEncodingFn) {
  auto resultType = emptyOp.getResult().getType().cast<RankedTensorType>();
  Optional<TensorEncoding> encoding = getEncoding(resultType);
  if (!encoding) {
    return rewriter.notifyMatchFailure(emptyOp,
                                       "result type does not have encoding");
  }
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(encoding.value());
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(
        emptyOp, "failed to find materialization info for result type");
  }
  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter, materializeEncodingInfo->innerTileSizes);
  SmallVector<OpFoldResult> newShape = PackOp::getResultShape(
      rewriter, emptyOp.getLoc(), emptyOp.getMixedSizes(), innerTileSizesOfr,
      materializeEncodingInfo->innerDimsPos,
      materializeEncodingInfo->outerDimsPerm);
  Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
      emptyOp.getLoc(), newShape, resultType.getElementType());
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
    MaterializeEncodingFn &materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(getTypeConverter())
            ->getMaterializeEncodingFn();
    // Pack op needs a padding value. Maybe that is an overkill. For now, just
    // use zero.
    auto packOp = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn);
    if (failed(packOp))
      return rewriter.notifyMatchFailure(encodingOp,
                                         "failed to convert to pack op");
    rewriter.replaceOp(encodingOp, packOp->getResults());
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
    MaterializeEncodingFn &materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(getTypeConverter())
            ->getMaterializeEncodingFn();
    auto unpackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn);
    if (failed(unpackOp))
      return rewriter.notifyMatchFailure(encodingOp,
                                         "failed to convert to unpack op");
    rewriter.replaceOp(encodingOp, unpackOp->getResults());
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
    MaterializeEncodingFn &materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp =
        lowerOpWithEncoding(rewriter, dpsOp, adaptor.getInputs(),
                            adaptor.getOutputs(), materializeEncodingFn);
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
    MaterializeEncodingFn &materializeEncodingFn =
        static_cast<MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, op, adaptor.getOperands(), materializeEncodingFn);
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
    RewritePatternSet patterns(context);
    MaterializeEncodingTypeConverter typeConverter(getPackOpInfoFromEncoding);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(patterns, target, typeConverter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops.
  {
    RewritePatternSet patterns(context);
    populateFoldIntoPackAndUnpackOpsPatterns(patterns);
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
    MaterializeEncodingTypeConverter &typeConverter) {

  // Add all patterns for converting from encoded type to the materialized type
  patterns.insert<MaterializeDPSOperation<linalg::FillOp>,
                  MaterializeDPSOperation<linalg::MatmulOp>,
                  MaterializeOperation<tensor::EmptyOp>,
                  SetEncodingOpToPackOpConversion,
                  UnsetEncodingOpToPackOpConversion>(typeConverter,
                                                     patterns.getContext());
  ::mlir::memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
}

std::unique_ptr<OperationPass<func::FuncOp>> createMaterializeEncodingPass() {
  return std::make_unique<MaterializeEncodingPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
