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

/// Given the `encoding` return (by value) the dimensions of the input that are
/// tiled (`innerDimsPos`), the tile size to use (`innerTileSizes`) and
/// permutation for the outer dimensions on the pack op (`outerDimsPerm`).
// TODO(ravishankarm): THis is currently hard-coded here for convenience. When
// used in IREE, this will be computed based on the architecture information in
// `hal.executable.variant`.
static LogicalResult getPackOpInfoFromEncoding(
    TensorEncoding encoding, SmallVector<int64_t> &innerDimsPos,
    SmallVector<int64_t> &innerTileSizes, SmallVector<int64_t> &outerDimsPerm) {
  switch (encoding) {
  case TensorEncoding::GEMM_LHS:
    innerDimsPos = {0, 1};
    innerTileSizes = {8, 4};
    outerDimsPerm = {};
    break;
  case TensorEncoding::GEMM_RHS:
    innerDimsPos = {0, 1};
    innerTileSizes = {4, 8};
    outerDimsPerm = {};
    break;
  case TensorEncoding::GEMM_RESULT:
    innerDimsPos = {0, 1};
    innerTileSizes = {8, 8};
    outerDimsPerm = {};
    break;
  case TensorEncoding::GEMM_RHS_TRANSPOSE:
    innerDimsPos = {1, 0};
    innerTileSizes = {8, 4};
    outerDimsPerm = {1, 0};
    break;
  default:
    return failure();
  }
  return success();
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
static RankedTensorType getMaterializedType(RankedTensorType tensorType) {
  Optional<TensorEncoding> encoding = getEncoding(tensorType);
  if (!encoding)
    return tensorType;
  SmallVector<int64_t> innerDimsPos, innerTileSizes, outerDimsPerm;
  if (failed(getPackOpInfoFromEncoding(encoding.value(), innerDimsPos,
                                       innerTileSizes, outerDimsPerm))) {
    return tensorType;
  }
  return PackOp::getPackedType(tensorType, innerTileSizes, innerDimsPos,
                               outerDimsPerm)
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
static FailureOr<PackOp> lowerSetEncodingOpToPackOp(RewriterBase &rewriter,
                                                    SetEncodingOp encodingOp,
                                                    Value source) {
  SmallVector<int64_t> innerDimsPos, innerTileSizes, outerDimsPerm;
  if (failed(getPackOpInfoFromEncoding(encodingOp.getResultTensorEncoding(),
                                       innerDimsPos, innerTileSizes,
                                       outerDimsPerm))) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled result encoding");
  }

  // Create `tensor.empty` operation for the result of the pack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> sourceDims = getDims(rewriter, loc, source);
  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter, innerTileSizes);
  SmallVector<OpFoldResult> resultDims =
      PackOp::getResultShape(rewriter, loc, sourceDims, innerTileSizesOfr,
                             innerDimsPos, outerDimsPerm);
  auto initTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultDims, encodingOp.getSourceType().getElementType());
  Optional<Value> paddingValue = getPaddingValue(source);
  return rewriter.create<PackOp>(loc, source, initTensor, innerDimsPos,
                                 innerTileSizesOfr, paddingValue,
                                 outerDimsPerm);
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// The source is taken as input so that these could be used with
/// `OpConversionPatterns`.
static FailureOr<UnPackOp>
lowerUnsetEncodingToUnpackOp(RewriterBase &rewriter, UnsetEncodingOp encodingOp,
                             Value packedValue) {
  SmallVector<int64_t> innerDimsPos, innerTileSizes, outerDimsPerm;
  if (failed(getPackOpInfoFromEncoding(encodingOp.getSourceTensorEncoding(),
                                       innerDimsPos, innerTileSizes,
                                       outerDimsPerm))) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled source encoding");
  }
  // Create an `tensor.empty` for the result of the unpack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> resultDims =
      getDims(rewriter, loc, encodingOp.getSource());
  auto initTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultDims, encodingOp.getResultType().getElementType());

  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter, innerTileSizes);
  return rewriter.create<UnPackOp>(loc, packedValue, initTensor, innerDimsPos,
                                   innerTileSizesOfr, outerDimsPerm);
}

namespace {

//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Convert `set_encoding` op to `pack` op.
struct SetEncodingOpToPackOpConversion
    : public OpConversionPattern<SetEncodingOp> {
  using OpConversionPattern<SetEncodingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Pack op needs a padding value. Maybe that is an overkill. For now, just
    // use zero.
    auto packOp =
        lowerSetEncodingOpToPackOp(rewriter, encodingOp, adaptor.getSource());
    if (failed(packOp))
      return rewriter.notifyMatchFailure(encodingOp,
                                         "failed to convert to pack op");
    rewriter.replaceOp(encodingOp, packOp->getResults());
    return success();
  }
};

/// Convert `unset_encoding` op to `unpack` op.
struct UnsetEncodingOpToPackOpConversion
    : public OpConversionPattern<UnsetEncodingOp> {
  using OpConversionPattern<UnsetEncodingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnsetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto unpackOp =
        lowerUnsetEncodingToUnpackOp(rewriter, encodingOp, adaptor.getSource());
    if (failed(unpackOp))
      return rewriter.notifyMatchFailure(encodingOp,
                                         "failed to convert to unpack op");
    rewriter.replaceOp(encodingOp, unpackOp->getResults());
    return success();
  }
};

/// Convert a linalg.matmul with
/// - lhs encoding of GEMM_LHS
/// - rhs encoding of GEMM_RHS_TRANSPOSE
/// - result encoding of GEMM_RESULT
/// to linalg.mmt4d op.
struct MaterializeTiledMatmul : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    rewriter.replaceOpWithNewOp<linalg::Mmt4DOp>(
        matmulOp,
        getMaterializedType(
            outputs[0]->get().getType().cast<RankedTensorType>()),
        adaptor.getInputs(), adaptor.getOutputs());
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Pass to materialize encoding
//===---------------------------------------------------------------------===//

struct MaterializeEncodingPass
    : public MaterializeEncodingBase<MaterializeEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    return;
  }

  void runOnOperation() override;
};

void MaterializeEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    TypeConverter typeConverter;
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

void populateMaterializeEncodingPatterns(RewritePatternSet &patterns,
                                         ConversionTarget &target,
                                         TypeConverter &typeConverter) {

  auto typeHasEncoding = [](Type t) -> bool {
    auto tensorType = t.dyn_cast<RankedTensorType>();
    return tensorType && tensorType.getEncoding();
  };
  auto valueHasEncoding = [&](Value v) -> bool {
    return typeHasEncoding(v.getType());
  };

  // Type converted is used to convert the unpacked tensor with tensor encoding
  // into a packed type.
  typeConverter.addConversion([&](RankedTensorType t) -> RankedTensorType {
    return getMaterializedType(t);
  });

  // Mark any operation that has operands/results with encoding as
  // illegal.
  auto hasOperandOrResultsWithEncoding = [&](Operation *op) {
    return llvm::any_of(op->getOperands(), valueHasEncoding) ||
           llvm::any_of(op->getResults(), valueHasEncoding);
  };
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return !hasOperandOrResultsWithEncoding(op); });

  // Add all patterns for converting from encoded type to the materialized type
  patterns.insert<MaterializeTiledMatmul, SetEncodingOpToPackOpConversion,
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
