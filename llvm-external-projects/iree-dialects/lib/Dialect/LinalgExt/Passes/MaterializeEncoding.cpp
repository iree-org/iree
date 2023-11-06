// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/EncodingUtils.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::LinalgExt;

//===---------------------------------------------------------------------===//
// Utility methods
//===---------------------------------------------------------------------===//

static EncodingAttr getEncodingAttr(RankedTensorType type) {
  return type.getEncoding().dyn_cast_or_null<EncodingAttr>();
}

static RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type) {
  auto encoding = getEncodingAttr(type);
  if (!encoding) {
    return type;
  }
  RankedTensorType originalType = type;
  if (auto originalTypeAttr = encoding.getOriginalType()) {
    originalType = originalTypeAttr.getValue().cast<RankedTensorType>();
  }
  return RankedTensorType::get(originalType.getShape(),
                               originalType.getElementType(), encoding);
}

static RankedTensorType dropEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

/// For a given tensor type with an encoding, return the materialized
/// type to use for it. If no encoding is set, then return the tensor type
/// itself.
static RankedTensorType
getMaterializedType(RankedTensorType tensorType,
                    MaterializeEncodingFn materializeEncodingFn) {
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);
  if (failed(materializeEncodingInfo)) {
    return dropEncoding(tensorType);
  }
  return tensor::PackOp::inferPackedType(
             getOriginalTypeWithEncoding(tensorType),
             materializeEncodingInfo->innerTileSizes,
             materializeEncodingInfo->innerDimsPos,
             materializeEncodingInfo->outerDimsPerm)
      .cast<RankedTensorType>();
}

static Operation *dropEncodingAndCloneOp(OpBuilder &builder, Operation *op,
                                         ValueRange convertedInputOperands,
                                         ValueRange convertedOutputOperands) {
  SmallVector<Value> operands;
  operands.append(convertedInputOperands.begin(), convertedInputOperands.end());
  operands.append(convertedOutputOperands.begin(),
                  convertedOutputOperands.end());
  return mlir::clone(
      builder, op,
      {dropEncoding(
          convertedOutputOperands[0].getType().cast<RankedTensorType>())},
      operands);
}

//===---------------------------------------------------------------------===//
// Methods to convert the encoding to parameters of the Pack operation
//===---------------------------------------------------------------------===//

/// Given the `encoding` return the `MaterializeEncodingInfo` to use for
/// materializing the pack op. This is mainly for testing. The configurations
/// are arbitrary values.
// TODO(hanchung): Move the implementation to Codegen/Common. This is currently
// hard-coded here for testing convenience. When used in IREE, this will be
// computed based on the architecture information in `hal.executable.variant`.
// A real implementation would return tile sizes that depend on at least the
// `tensorType`'s element type (e.g. different tile sizes for i8 vs f32, because
// the SIMD instructions may have different shapes).
// Moreover, in a real implementation, the tile sizes would typically also
// depend on target information. This is demonstrated in
// iree/compiler/src/iree/compiler/Codegen/Common/MaterializeEncodingPass.cpp
static FailureOr<MaterializeEncodingInfo>
chooseEncodingInfo(RankedTensorType tensorType) {
  auto encoding = getEncodingAttr(tensorType);
  if (!encoding)
    return failure();

  auto user = encoding.getUser().getValue();
  auto role = encoding.getRole().getValue();
  // Below is for testing purpose. It only materialize for f32 cases.
  switch (user) {
  case EncodingUser::MATMUL:
  case EncodingUser::BATCH_MATMUL:
    if (tensorType.getElementType().isF32()) {
      return getEncodingInfoForMatmul(user, role, /*tileParams=*/{8, 8, 4});
    }
  }
  return failure();
}

//===---------------------------------------------------------------------===//
// Methods to convert `set_encoding` and `unset_encoding` operations
// to `pack` and `unpack` operations respectively.
//===---------------------------------------------------------------------===//

/// Utility method to get the optional padding value to use with pack operation
/// if source is defined using a `tensor.pad` operation. Note `source` is
/// passed by reference. It is updated to use the source of the pad operation.
static std::optional<Value> getPaddingValue(Value &source) {
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
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, resultType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  auto encoding = getEncodingAttr(resultType);
  if (!encoding)
    return failure();
  std::optional<Value> paddingValue = getPaddingValue(source);
  SmallVector<OpFoldResult> sourceDims = getDims(rewriter, loc, source);
  SmallVector<OpFoldResult> resultDims = tensor::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr,
      materializeEncodingInfo->innerDimsPos,
      materializeEncodingInfo->outerDimsPerm);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  resultType.getElementType());
  return rewriter.create<tensor::PackOp>(
      loc, source, emptyOp, materializeEncodingInfo->innerDimsPos,
      *innerTileSizesOfr, paddingValue, materializeEncodingInfo->outerDimsPerm);
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

static FailureOr<SmallVector<Value>> lowerUpperBoundTileSizeOpToConstants(
    RewriterBase &rewriter, UpperBoundTileSizeOp upperBoundTileSizeOp,
    MaterializeEncodingFn materializeEncodingFn) {
  Location loc = upperBoundTileSizeOp.getLoc();
  RankedTensorType tensorType = upperBoundTileSizeOp.getTensorType();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(upperBoundTileSizeOp,
                                       "unhandled source encoding");
  }
  ArrayRef<int64_t> innerTileSizes = materializeEncodingInfo->innerTileSizes;
  ArrayRef<int64_t> innerDimsPos = materializeEncodingInfo->innerDimsPos;
  SmallVector<Value> results(tensorType.getRank());
  for (unsigned i = 0; i < innerTileSizes.size(); ++i) {
    int64_t tileSize = innerTileSizes[i];
    if (ShapedType::isDynamic(tileSize)) {
      tileSize = 16;
    }
    results[innerDimsPos[i]] =
        rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
  }
  // For the dims that have no inner tiles, use 1 as tile size to avoid padding.
  for (unsigned i = 0; i < results.size(); ++i) {
    if (!results[i]) {
      results[i] = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    }
  }
  return results;
}

/// Utility method to convert from `linalg.matmul` with
/// - lhs encoding with role=LHS
/// - rhs encoding with role=RHS
/// - result encoding with role=RESULT
/// to linalg.mmt4d op.
static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::MatmulOp matmulOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    MaterializeEncodingFn materializeEncodingFn, MaterializeEncodingValueFn) {
  if (!matmulOp.hasTensorSemantics())
    return failure();
  auto inputs = matmulOp.getDpsInputOperands();
  auto outputs = matmulOp.getDpsInits();
  auto lhsEncoding =
      getEncodingAttr(inputs[0]->get().getType().cast<RankedTensorType>());
  auto rhsEncoding =
      getEncodingAttr(inputs[1]->get().getType().cast<RankedTensorType>());
  auto resultEncoding =
      getEncodingAttr(outputs[0].getType().cast<RankedTensorType>());
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }
  if (!isMatmulEncodingUser(lhsEncoding.getUser().getValue()) ||
      !isMatmulEncodingUser(rhsEncoding.getUser().getValue()) ||
      !isMatmulEncodingUser(resultEncoding.getUser().getValue()) ||
      lhsEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::LHS ||
      rhsEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::RHS ||
      resultEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::RESULT) {
    return failure();
  }

  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(getOriginalTypeWithEncoding(
          matmulOp.getResultTypes()[0].cast<RankedTensorType>()));
  Operation *result;
  if (failed(materializeEncodingInfo)) {
    result = dropEncodingAndCloneOp(rewriter, matmulOp, convertedInputOperands,
                                    convertedOutputOperands);
  } else {
    result = rewriter.create<linalg::Mmt4DOp>(
        matmulOp.getLoc(), convertedOutputOperands[0].getType(),
        convertedInputOperands, convertedOutputOperands);
  }
  return result;
}

/// Utility method to convert from `linalg.batch_matmul` with
/// - lhs encoding with user=BATCH_MATMUL_*, role=LHS
/// - rhs encoding with user=BATCH_MATMUL_*, role=RHS
/// - result encoding with user=BATCH_MATMUL_*, role=RESULT
/// to linalg.batch_mmt4d op.
static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::BatchMatmulOp batchMatmulOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    MaterializeEncodingFn materializeEncodingFn, MaterializeEncodingValueFn) {
  if (!batchMatmulOp.hasTensorSemantics())
    return failure();
  auto inputs = batchMatmulOp.getDpsInputOperands();
  auto outputs = batchMatmulOp.getDpsInits();
  auto lhsEncoding =
      getEncodingAttr(inputs[0]->get().getType().cast<RankedTensorType>());
  auto rhsEncoding =
      getEncodingAttr(inputs[1]->get().getType().cast<RankedTensorType>());
  auto resultEncoding =
      getEncodingAttr(outputs[0].getType().cast<RankedTensorType>());
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }

  if (!isBatchMatmulEncodingUser(lhsEncoding.getUser().getValue()) ||
      !isBatchMatmulEncodingUser(rhsEncoding.getUser().getValue()) ||
      !isBatchMatmulEncodingUser(resultEncoding.getUser().getValue()) ||
      lhsEncoding.getRole().getValue() != EncodingRole::LHS ||
      rhsEncoding.getRole().getValue() != EncodingRole::RHS ||
      resultEncoding.getRole().getValue() != EncodingRole::RESULT) {
    return failure();
  }
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(getOriginalTypeWithEncoding(
          batchMatmulOp.getResultTypes()[0].cast<RankedTensorType>()));
  Operation *result;
  if (failed(materializeEncodingInfo)) {
    result =
        dropEncodingAndCloneOp(rewriter, batchMatmulOp, convertedInputOperands,
                               convertedOutputOperands);
  } else {
    result = rewriter.create<linalg::BatchMmt4DOp>(
        batchMatmulOp.getLoc(), convertedOutputOperands[0].getType(),
        convertedInputOperands, convertedOutputOperands);
  }
  return result;
}

/// Utility method to convert from `linalg.fill` on `tensor` type with
/// encoding to fill of the materialized type
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
  auto resultType = getOriginalTypeWithEncoding(
      emptyOp->getResultTypes()[0].cast<RankedTensorType>());
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(resultType);
  Location loc = emptyOp.getLoc();
  if (failed(materializeEncodingInfo)) {
    Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, emptyOp.getMixedSizes(), resultType.getElementType());
    return newEmptyOp;
  }

  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, resultType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        emptyOp, "failed to generate runtime tile size query");
  }
  SmallVector<OpFoldResult> sourceDims = emptyOp.getMixedSizes();
  (void)foldDynamicIndexList(sourceDims);
  SmallVector<OpFoldResult> newShape =
      PackOp::getResultShape(rewriter, loc, sourceDims, *innerTileSizesOfr,
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
        static_cast<const MaterializeEncodingTypeConverter *>(
            getTypeConverter())
            ->getMaterializeEncodingFn();
    auto packOp = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(packOp)) {
      Value result = adaptor.getSource();
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      if (targetType != result.getType()) {
        result = rewriter.create<tensor::CastOp>(encodingOp.getLoc(),
                                                 targetType, result);
      }
      rewriter.replaceOp(encodingOp, result);
      return success();
    }
    rewriter.replaceOp(encodingOp, packOp->getResult());
    return success();
  }
};

/// Convert `unset_encoding` op to `unpack` op.
struct UnsetEncodingOpToUnPackOpConversion
    : public OpMaterializeEncodingPattern<UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(UnsetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    auto unpackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(unpackOp)) {
      Value result = adaptor.getSource();
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      if (targetType != result.getType()) {
        result = rewriter.create<tensor::CastOp>(encodingOp.getLoc(),
                                                 targetType, result);
      }
      rewriter.replaceOp(encodingOp, result);
      return success();
    }
    rewriter.replaceOp(encodingOp, unpackOp->getResult());
    return success();
  }
};

/// Convert `upper_bound_tile_size` op to `constant` op. If the
/// `materializeEncodingFn` returns a failure, the pattern will materialize it
/// to the same shape.
struct UpperBoundTileSizeToConstantOpConversion
    : public OpRewritePattern<UpperBoundTileSizeOp> {
  UpperBoundTileSizeToConstantOpConversion(
      MLIRContext *context, MaterializeEncodingFn materializeEncodingFn)
      : OpRewritePattern<UpperBoundTileSizeOp>(context),
        materializeEncodingFn(materializeEncodingFn) {}

  LogicalResult matchAndRewrite(UpperBoundTileSizeOp upperBoundTileSizeOp,
                                PatternRewriter &rewriter) const override {

    auto constants = lowerUpperBoundTileSizeOpToConstants(
        rewriter, upperBoundTileSizeOp, materializeEncodingFn);
    if (failed(constants)) {
      SmallVector<Value> results(upperBoundTileSizeOp.getNumResults(),
                                 rewriter.create<arith::ConstantIndexOp>(
                                     upperBoundTileSizeOp.getLoc(), 1));
      rewriter.replaceOp(upperBoundTileSizeOp, results);
      return success();
    }
    rewriter.replaceOp(upperBoundTileSizeOp, *constants);
    return success();
  }

  MaterializeEncodingFn materializeEncodingFn;
};

/// Generic pattern to convert operaiton that is in Destination Passing Style.
template <typename OpTy>
struct MaterializeDPSOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy dpsOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
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
        static_cast<const MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, op, adaptor.getOperands(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(convertedOp))
      return failure();

    SmallVector<Value> replacements;
    for (auto [type, res] : llvm::zip_equal(
             op->getResultTypes(), convertedOp.value()->getResults())) {
      Type targetType = this->getTypeConverter()->convertType(type);
      if (targetType == res.getType()) {
        replacements.push_back(res);
      } else {
        replacements.push_back(
            rewriter.create<tensor::CastOp>(op.getLoc(), targetType, res));
      }
    }
    rewriter.replaceOp(op, replacements);
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

  // Add all patterns for converting from encoded type to the materialized
  // type
  patterns.insert<MaterializeDPSOperation<linalg::FillOp>,
                  MaterializeDPSOperation<linalg::MatmulOp>,
                  MaterializeDPSOperation<linalg::BatchMatmulOp>,
                  MaterializeOperation<tensor::EmptyOp>,
                  SetEncodingOpToPackOpConversion,
                  UnsetEncodingOpToUnPackOpConversion>(
      patterns.getContext(), typeConverter, materializeEncodingValueFn);
  ::mlir::memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
}

void populateMaterializeUpperBoundTileSizePatterns(
    RewritePatternSet &patterns, MaterializeEncodingFn materializeEncodingFn) {
  patterns.insert<UpperBoundTileSizeToConstantOpConversion>(
      patterns.getContext(), materializeEncodingFn);
}

std::unique_ptr<OperationPass<func::FuncOp>> createMaterializeEncodingPass() {
  return std::make_unique<MaterializeEncodingPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
